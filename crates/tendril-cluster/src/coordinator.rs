//! The coordinator: owns the model files, admits machines, plans, loads the
//! pipeline and drives generation for every request.

use crate::agent::{connect_retry, pump};
use crate::calibration::Calibration;
use crate::proto::{Msg, NextHop, Payload, SampleSetup, StageTime, TensorEntry, PROTOCOL};
use crate::shard::{model_key, stage_tensors};
use crate::worker::StageWorker;
use crate::{token, wire};
use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tendril_core::cluster::{Cluster, Link};
use tendril_core::hardware::NodeProfile;
use tendril_core::model::{ModelSpec, Quant};
use tendril_core::planner::{self, Goal, Plan, PlanOptions, PlanResult, Workload};
use tendril_core::units::{fmt_ms, Bytes};
use tendril_engine::config::ModelConfig;
use tendril_engine::device::{activation_dtype, device_from_name, device_label};
use tendril_engine::generate::{prefill_chunk, FinishReason, StopMatcher, Timing};
use tendril_engine::linear::WeightFormat;
use tendril_engine::model::{LoadOptions, Stage, StageSpec};
use tendril_engine::sampler::SamplingParams;
use tendril_engine::tokenizer::{Detokenizer, Tok};
use tendril_engine::weights::WeightStore;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, mpsc, Notify, OwnedSemaphorePermit, RwLock, Semaphore};

#[derive(Clone, Debug)]
pub struct ServeOptions {
    pub model_dir: PathBuf,
    pub model_name: String,
    pub format: WeightFormat,
    pub context: usize,
    pub concurrency: usize,
    pub goal: Goal,
    pub safety: f64,
    pub control_port: u16,
    pub data_port: u16,
    pub token: String,
    /// Device for this machine's stage ("auto", "cpu", "metal", "cuda").
    pub device: String,
    /// Let this machine run part of the model.
    pub use_local: bool,
    pub link: Option<Link>,
    pub name: Option<String>,
    /// Use at least this many machines.
    pub min_stages: usize,
    /// Cap on memory Tendril may use on this machine.
    pub max_memory: Option<Bytes>,
}

/// Human-readable log of what the cluster is doing.
#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub at: String,
    pub level: &'static str,
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct StageStatus {
    pub node: String,
    pub components: String,
    pub layers: (u64, u64),
    pub phase: String,
    pub done: u64,
    pub total: u64,
    pub device: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Status {
    Starting,
    /// No feasible plan with the machines present.
    Waiting {
        reason: String,
        advice: Vec<String>,
    },
    Loading {
        plan: String,
        stages: Vec<StageStatus>,
    },
    Ready {
        plan: String,
        stages: Vec<StageStatus>,
        predicted_tps: f64,
        since_ms: u64,
    },
    Failed {
        error: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct NodeInfo {
    pub id: u32,
    pub name: String,
    pub chip: String,
    pub backend: String,
    pub usable: String,
    pub usable_bytes: u64,
    pub total: String,
    pub address: String,
    pub rtt_ms: Option<f64>,
    pub bandwidth_gbps: Option<f64>,
    pub local: bool,
    pub role: String,
    pub on_battery: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Metrics {
    pub requests: u64,
    pub active: usize,
    pub queued: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub errors: u64,
    pub recent: VecDeque<RequestRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RequestRecord {
    pub at: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: f64,
    pub decode_tps: f64,
    pub total_ms: f64,
    pub finish: String,
}

/// Generation output streamed to the caller.
#[derive(Debug)]
pub enum GenOut {
    Text(String),
    Done {
        reason: FinishReason,
        timing: Timing,
    },
    Error(String),
}

#[derive(Clone, Debug)]
pub struct GenRequest {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    /// Keep generating past end-of-sequence tokens (benchmarks).
    pub ignore_eos: bool,
}

#[derive(Debug)]
pub enum ServeError {
    NotReady(String),
    BadRequest(String),
    Busy(String),
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServeError::NotReady(s) | ServeError::BadRequest(s) | ServeError::Busy(s) => {
                f.write_str(s)
            }
        }
    }
}

struct RemoteNode {
    tx: mpsc::Sender<Msg>,
    peer_ip: IpAddr,
    local_ip: IpAddr,
    data_port: u16,
    /// Load-time replies (NeedWeights, progress, ready, failed).
    inbox: StdMutex<Option<mpsc::UnboundedSender<Msg>>>,
    pongs: StdMutex<HashMap<u64, Instant>>,
}

struct Node {
    id: u32,
    profile: NodeProfile,
    remote: Option<Arc<RemoteNode>>,
    rtt_ms: Option<f64>,
    bandwidth_gbps: Option<f64>,
}

struct StageSlot {
    node_id: u32,
    spec: StageSpec,
}

struct Pipeline {
    epoch: u64,
    plan: Plan,
    slots: Vec<StageSlot>,
    entry: Entry,
    permits: Arc<Semaphore>,
    /// Keeps the in-process stage alive.
    _local: Option<Arc<StageWorker>>,
    ready_at: Instant,
    telemetry: StdMutex<Telemetry>,
}

enum Entry {
    Local(Arc<StageWorker>),
    Remote(mpsc::UnboundedSender<Msg>),
}

impl Entry {
    fn send(&self, m: Msg) -> bool {
        match self {
            Entry::Local(w) => w.submit(m),
            Entry::Remote(tx) => tx.send(m).is_ok(),
        }
    }
}

enum SeqEvent {
    Token(u32, Vec<StageTime>),
    Error(String),
}

/// Measured per-token timings of the running pipeline (decode steps only).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Telemetry {
    pub samples: u64,
    /// Coordinator-observed time per decode step, ms.
    pub step_ms: f64,
    /// Per-stage compute time per decode step, ms (stage order).
    pub compute_ms: Vec<f64>,
    /// Per-stage queueing delay, ms.
    pub queue_ms: Vec<f64>,
    /// Step time not spent computing: network, serialization, scheduling, ms.
    pub transfer_ms: f64,
}

impl Telemetry {
    fn record(&mut self, step_ms: f64, trace: &[StageTime], stages: usize) {
        if self.compute_ms.len() != stages {
            self.compute_ms = vec![0.0; stages];
            self.queue_ms = vec![0.0; stages];
        }
        self.samples += 1;
        // Running mean for the first samples, then an exponential average.
        let a = (1.0 / self.samples as f64).max(0.02);
        let mix = |old: &mut f64, new: f64| *old += a * (new - *old);
        mix(&mut self.step_ms, step_ms);
        let mut busy = 0.0;
        for t in trace {
            let i = t.stage as usize;
            if i < stages {
                mix(&mut self.compute_ms[i], t.compute_us as f64 / 1000.0);
                mix(&mut self.queue_ms[i], t.queue_us as f64 / 1000.0);
                busy += (t.compute_us + t.queue_us) as f64 / 1000.0;
            }
        }
        mix(&mut self.transfer_ms, (step_ms - busy).max(0.0));
    }
}

pub struct Inner {
    pub opts: ServeOptions,
    pub cfg: Arc<ModelConfig>,
    config_json: String,
    pub tok: Arc<Tok>,
    ws: Arc<WeightStore>,
    model_key: String,
    pub spec: ModelSpec,
    psk: [u8; 32],
    data_port: AtomicUsize,
    nodes: StdMutex<Vec<Node>>,
    next_node: AtomicU64,
    status: StdMutex<Status>,
    events: broadcast::Sender<Event>,
    history: StdMutex<VecDeque<Event>>,
    pipeline: RwLock<Option<Arc<Pipeline>>>,
    router: StdMutex<HashMap<u64, mpsc::UnboundedSender<SeqEvent>>>,
    local_worker: StdMutex<Option<(u64, Arc<StageWorker>)>>,
    next_seq: AtomicU64,
    epoch: AtomicU64,
    metrics: StdMutex<Metrics>,
    replan: Notify,
    last_plan: StdMutex<Option<PlanResult>>,
    started: Instant,
    calibration: StdMutex<Calibration>,
}

#[derive(Clone)]
pub struct Coordinator {
    pub inner: Arc<Inner>,
}

fn now() -> String {
    clock()
}

/// Local wall-clock time as HH:MM:SS.
pub fn clock() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

impl Coordinator {
    /// Open the model and start listening. Planning starts immediately.
    pub async fn start(opts: ServeOptions) -> Result<Coordinator> {
        let dir = opts.model_dir.clone();
        let cfg = Arc::new(ModelConfig::from_file(&dir.join("config.json"))?);
        let config_json = std::fs::read_to_string(dir.join("config.json"))?;
        let tok = Arc::new(Tok::from_dir(&dir, &cfg.eos_token_ids, cfg.bos_token_id)?);
        let ws = Arc::new(WeightStore::open_dir(&dir)?);
        let key = model_key(&config_json, &ws)?;
        let mut spec = tendril_core::model::source::inspect_local_dir(&dir)?;
        spec.id = opts.model_name.clone();
        let spec = match opts.format {
            WeightFormat::Native => spec,
            WeightFormat::Q8_0 => spec.with_repr(Quant::Q8_0),
            WeightFormat::Q4K => spec.with_repr(Quant::Q4K),
            WeightFormat::Q6K => spec.with_repr(Quant::Q6K),
        };
        let (events, _) = broadcast::channel(256);
        let inner = Arc::new(Inner {
            psk: token::psk(&opts.token),
            opts,
            cfg,
            config_json,
            tok,
            ws,
            model_key: key,
            spec,
            data_port: AtomicUsize::new(0),
            nodes: StdMutex::new(Vec::new()),
            next_node: AtomicU64::new(1),
            status: StdMutex::new(Status::Starting),
            events,
            history: StdMutex::new(VecDeque::new()),
            pipeline: RwLock::new(None),
            router: StdMutex::new(HashMap::new()),
            local_worker: StdMutex::new(None),
            next_seq: AtomicU64::new(1),
            epoch: AtomicU64::new(0),
            metrics: StdMutex::new(Metrics::default()),
            replan: Notify::new(),
            last_plan: StdMutex::new(None),
            started: Instant::now(),
            calibration: StdMutex::new(Calibration::load()),
        });
        let c = Coordinator { inner };

        // This machine.
        if c.inner.opts.use_local {
            let mut p = tendril_core::hardware::detect_local(true);
            if let Some(n) = &c.inner.opts.name {
                p.name = n.clone();
            }
            let dev = device_from_name(&c.inner.opts.device)?;
            if matches!(dev, candle_core::Device::Cpu) && p.backend != tendril_core::Backend::Cpu {
                p.backend = tendril_core::Backend::Cpu;
                p.recompute_usable();
            }
            apply_memory_cap(&mut p, c.inner.opts.max_memory);
            c.add_node(p, None);
        }

        let control = TcpListener::bind(("0.0.0.0", c.inner.opts.control_port))
            .await
            .with_context(|| {
                format!(
                    "cannot listen on port {} (is another Tendril running?)",
                    c.inner.opts.control_port
                )
            })?;
        let data = TcpListener::bind(("0.0.0.0", c.inner.opts.data_port)).await?;
        c.inner
            .data_port
            .store(data.local_addr()?.port() as usize, Ordering::Relaxed);
        tokio::spawn(c.clone().control_loop(control));
        tokio::spawn(c.clone().data_loop(data));
        tokio::spawn(c.clone().planner_loop());
        tokio::spawn(c.clone().heartbeat_loop());
        c.inner.replan.notify_one();
        Ok(c)
    }

    pub fn control_port(&self) -> u16 {
        self.inner.opts.control_port
    }

    pub fn event(&self, level: &'static str, text: impl Into<String>) {
        let e = Event {
            at: now(),
            level,
            text: text.into(),
        };
        tracing::info!("{}", e.text);
        {
            let mut h = self.inner.history.lock().unwrap();
            h.push_back(e.clone());
            while h.len() > 200 {
                h.pop_front();
            }
        }
        let _ = self.inner.events.send(e);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.inner.events.subscribe()
    }

    pub fn history(&self) -> Vec<Event> {
        self.inner.history.lock().unwrap().iter().cloned().collect()
    }

    pub fn status(&self) -> Status {
        self.inner.status.lock().unwrap().clone()
    }

    fn set_status(&self, s: Status) {
        *self.inner.status.lock().unwrap() = s;
    }

    pub fn metrics(&self) -> Metrics {
        self.inner.metrics.lock().unwrap().clone()
    }

    pub fn last_plan(&self) -> Option<PlanResult> {
        self.inner.last_plan.lock().unwrap().clone()
    }

    pub fn uptime(&self) -> Duration {
        self.inner.started.elapsed()
    }

    pub fn nodes(&self) -> Vec<NodeInfo> {
        let roles: HashMap<u32, String> = match self.inner.pipeline.try_read() {
            Ok(g) => g
                .as_ref()
                .map(|p| {
                    p.slots
                        .iter()
                        .map(|s| {
                            let mut parts = vec![];
                            if s.spec.embed {
                                parts.push("embeddings".to_string());
                            }
                            parts.push(format!(
                                "layers {}–{}",
                                s.spec.layer_start,
                                s.spec.layer_end.saturating_sub(1)
                            ));
                            if s.spec.head {
                                parts.push("head".to_string());
                            }
                            (s.node_id, parts.join(" + "))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => HashMap::new(),
        };
        self.inner
            .nodes
            .lock()
            .unwrap()
            .iter()
            .map(|n| NodeInfo {
                id: n.id,
                name: n.profile.name.clone(),
                chip: n.profile.chip.clone(),
                backend: n.profile.backend.label().to_string(),
                usable: n.profile.usable_memory.to_string(),
                usable_bytes: n.profile.usable_memory.0,
                total: n.profile.total_memory.to_string(),
                address: n
                    .remote
                    .as_ref()
                    .map(|r| format!("{}", r.peer_ip))
                    .unwrap_or_else(|| "this machine".into()),
                rtt_ms: n.rtt_ms,
                bandwidth_gbps: n.bandwidth_gbps,
                local: n.remote.is_none(),
                role: roles.get(&n.id).cloned().unwrap_or_else(|| "idle".into()),
                on_battery: n.profile.on_battery,
            })
            .collect()
    }

    fn add_node(&self, mut profile: NodeProfile, remote: Option<Arc<RemoteNode>>) -> (u32, String) {
        let mut nodes = self.inner.nodes.lock().unwrap();
        let base = profile.name.clone();
        let mut name = base.clone();
        let mut i = 2;
        while nodes.iter().any(|n| n.profile.name == name) {
            name = format!("{base}-{i}");
            i += 1;
        }
        profile.name = name.clone();
        let id = self.inner.next_node.fetch_add(1, Ordering::Relaxed) as u32;
        nodes.push(Node {
            id,
            profile,
            remote,
            rtt_ms: None,
            bandwidth_gbps: None,
        });
        (id, name)
    }

    // ------------------------------------------------------------------
    // Membership

    async fn control_loop(self, listener: TcpListener) {
        loop {
            let Ok((s, _)) = listener.accept().await else {
                continue;
            };
            let c = self.clone();
            tokio::spawn(async move {
                if let Err(e) = c.handle_agent(s).await {
                    tracing::debug!("agent session ended: {e:#}");
                }
            });
        }
    }

    async fn handle_agent(&self, s: tokio::net::TcpStream) -> Result<()> {
        let local_ip = s.local_addr()?.ip();
        let conn = match wire::accept(s, &self.inner.psk).await {
            Ok(c) => c,
            Err(e) => {
                self.event("warn", format!("Rejected a connection: {e:#}"));
                return Err(e);
            }
        };
        let peer_ip = conn.peer.ip();
        let (mut reader, mut writer) = (conn.reader, conn.writer);
        let (profile, data_port) = match tokio::time::timeout(
            Duration::from_secs(10),
            reader.recv(),
        )
        .await??
        {
            Msg::Hello {
                protocol,
                profile,
                data_port,
                version,
                ..
            } => {
                if protocol != PROTOCOL {
                    let reason = format!(
                        "protocol mismatch: this coordinator speaks v{PROTOCOL} (tendril {}), you run tendril {version}",
                        env!("CARGO_PKG_VERSION")
                    );
                    writer
                        .send(&Msg::Reject {
                            reason: reason.clone(),
                        })
                        .await?;
                    bail!(reason);
                }
                (profile, data_port)
            }
            other => bail!("expected Hello, got {other:?}"),
        };
        let (tx, mut rx) = mpsc::channel::<Msg>(16);
        let remote = Arc::new(RemoteNode {
            tx: tx.clone(),
            peer_ip,
            local_ip,
            data_port,
            inbox: StdMutex::new(None),
            pongs: StdMutex::new(HashMap::new()),
        });
        let (id, name) = self.add_node(profile.clone(), Some(remote.clone()));
        writer
            .send(&Msg::Welcome {
                node_id: id,
                name: name.clone(),
                cluster: self.inner.opts.model_name.clone(),
            })
            .await?;
        let writer_task = tokio::spawn(async move {
            while let Some(m) = rx.recv().await {
                if writer.send(&m).await.is_err() {
                    break;
                }
            }
        });

        // Measure the link: a few pings, then a 4 MiB transfer.
        let measure = {
            let c = self.clone();
            let remote = remote.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                for i in 0..5u64 {
                    remote
                        .pongs
                        .lock()
                        .unwrap()
                        .insert(1000 + i, Instant::now());
                    let _ = remote.tx.send(Msg::Ping { t: 1000 + i }).await;
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                remote
                    .pongs
                    .lock()
                    .unwrap()
                    .insert(u64::MAX, Instant::now());
                let _ = remote
                    .tx
                    .send(Msg::Probe {
                        data: vec![0x5a; 4 << 20],
                    })
                    .await;
                let _ = c;
            })
        };

        self.event(
            "ok",
            format!(
                "{name} joined — {} · {} backend · {} for models",
                profile.chip,
                profile.backend.label(),
                profile.usable_memory
            ),
        );
        let mut rtts: Vec<f64> = Vec::new();
        let result: Result<()> = loop {
            let msg = match tokio::time::timeout(Duration::from_secs(15), reader.recv()).await {
                Ok(Ok(m)) => m,
                Ok(Err(e)) => break Err(e),
                Err(_) => break Err(anyhow!("stopped answering heartbeats")),
            };
            match msg {
                Msg::Pong { t } => {
                    if let Some(sent) = remote.pongs.lock().unwrap().remove(&t) {
                        let ms = sent.elapsed().as_secs_f64() * 1000.0;
                        rtts.push(ms);
                        if rtts.len() > 20 {
                            rtts.remove(0);
                        }
                        let mut sorted = rtts.clone();
                        sorted.sort_by(|a, b| a.total_cmp(b));
                        let median = sorted[sorted.len() / 2];
                        self.update_node(id, |n| n.rtt_ms = Some(median));
                    }
                }
                Msg::ProbeAck { len } => {
                    if let Some(sent) = remote.pongs.lock().unwrap().remove(&u64::MAX) {
                        let secs = sent.elapsed().as_secs_f64();
                        let rtt = self.node_rtt(id).unwrap_or(0.0) / 1000.0;
                        let gbps = (len as f64 * 8.0) / (secs - rtt / 2.0).max(1e-4) / 1e9;
                        self.update_node(id, |n| n.bandwidth_gbps = Some(gbps));
                        self.event(
                            "info",
                            format!(
                                "Link to {name}: {} round trip, {:.1} Gb/s measured",
                                fmt_ms(self.node_rtt(id).unwrap_or(0.0)),
                                gbps
                            ),
                        );
                        self.inner.replan.notify_one();
                    }
                }
                m @ (Msg::NeedWeights { .. }
                | Msg::LoadProgress { .. }
                | Msg::StageReady { .. }
                | Msg::LoadFailed { .. }) => {
                    if let Some(inbox) = remote.inbox.lock().unwrap().as_ref() {
                        let _ = inbox.send(m);
                    }
                }
                _ => {}
            }
        };
        measure.abort();
        writer_task.abort();
        self.remove_node(
            id,
            &name,
            result
                .as_ref()
                .err()
                .map(|e| format!("{e:#}"))
                .unwrap_or_default(),
        )
        .await;
        result
    }

    fn update_node(&self, id: u32, f: impl FnOnce(&mut Node)) {
        if let Some(n) = self
            .inner
            .nodes
            .lock()
            .unwrap()
            .iter_mut()
            .find(|n| n.id == id)
        {
            f(n);
        }
    }

    fn node_rtt(&self, id: u32) -> Option<f64> {
        self.inner
            .nodes
            .lock()
            .unwrap()
            .iter()
            .find(|n| n.id == id)
            .and_then(|n| n.rtt_ms)
    }

    async fn remove_node(&self, id: u32, name: &str, why: String) {
        self.inner.nodes.lock().unwrap().retain(|n| n.id != id);
        let in_use = self
            .inner
            .pipeline
            .read()
            .await
            .as_ref()
            .is_some_and(|p| p.slots.iter().any(|s| s.node_id == id));
        if in_use {
            self.event("error", format!("{name} left the cluster ({why}) — it was running part of the model; replanning"));
            self.teardown(&format!("{name} disconnected")).await;
        } else {
            self.event("warn", format!("{name} left the cluster"));
        }
        self.inner.replan.notify_one();
    }

    async fn heartbeat_loop(self) {
        let mut t = 1u64;
        loop {
            tokio::time::sleep(Duration::from_secs(3)).await;
            let remotes: Vec<Arc<RemoteNode>> = self
                .inner
                .nodes
                .lock()
                .unwrap()
                .iter()
                .filter_map(|n| n.remote.clone())
                .collect();
            for r in remotes {
                t += 1;
                r.pongs.lock().unwrap().insert(t, Instant::now());
                let _ = r.tx.try_send(Msg::Ping { t });
            }
        }
    }

    /// Fail all in-flight requests and drop the pipeline.
    async fn teardown(&self, why: &str) {
        let old = self.inner.pipeline.write().await.take();
        if let Some(p) = old {
            let routes: Vec<_> = self.inner.router.lock().unwrap().drain().collect();
            for (_, tx) in routes {
                let _ = tx.send(SeqEvent::Error(format!(
                    "the cluster lost a machine mid-request ({why}); please retry"
                )));
            }
            for s in &p.slots {
                if let Some(r) = self.remote(s.node_id) {
                    let _ = r.tx.try_send(Msg::Unload { epoch: p.epoch });
                }
            }
        }
        *self.inner.local_worker.lock().unwrap() = None;
    }

    fn remote(&self, id: u32) -> Option<Arc<RemoteNode>> {
        self.inner
            .nodes
            .lock()
            .unwrap()
            .iter()
            .find(|n| n.id == id)
            .and_then(|n| n.remote.clone())
    }

    // ------------------------------------------------------------------
    // Planning

    fn cluster_snapshot(&self) -> (Cluster, Vec<u32>) {
        let nodes = self.inner.nodes.lock().unwrap();
        let cal = self.inner.calibration.lock().unwrap();
        let profiles: Vec<NodeProfile> = nodes
            .iter()
            .map(|n| {
                let mut p = n.profile.clone();
                cal.apply(&mut p);
                p
            })
            .collect();
        drop(cal);
        let ids: Vec<u32> = nodes.iter().map(|n| n.id).collect();
        let default = self
            .inner
            .opts
            .link
            .clone()
            .unwrap_or_else(|| Link::preset("gbe").unwrap());
        let mut c = Cluster::new(profiles, default.clone());
        if self.inner.opts.link.is_none() {
            let link_of = |n: &Node| -> Option<Link> {
                match (n.remote.as_ref(), n.rtt_ms, n.bandwidth_gbps) {
                    (None, _, _) => Some(Link::preset("local").unwrap()),
                    (Some(_), Some(rtt), Some(bw)) => Some(Link {
                        name: "measured".into(),
                        bandwidth_gbps: bw,
                        rtt_ms: rtt,
                        measured: true,
                    }),
                    _ => None,
                }
            };
            for i in 0..nodes.len() {
                for j in i + 1..nodes.len() {
                    let (a, b) = (&nodes[i], &nodes[j]);
                    // Links are measured to the coordinator; a remote↔remote hop is
                    // estimated from the slower of the two.
                    let l = match (a.remote.is_none(), b.remote.is_none()) {
                        (true, _) => link_of(b),
                        (_, true) => link_of(a),
                        _ => match (link_of(a), link_of(b)) {
                            (Some(x), Some(y)) => Some(Link {
                                name: "estimated".into(),
                                bandwidth_gbps: x.bandwidth_gbps.min(y.bandwidth_gbps),
                                rtt_ms: x.rtt_ms.max(y.rtt_ms),
                                measured: false,
                            }),
                            _ => None,
                        },
                    };
                    if let Some(l) = l {
                        c.set_link(&a.profile.name, &b.profile.name, l);
                    }
                }
            }
        }
        (c, ids)
    }

    fn workload(&self) -> Workload {
        let mut w = Workload::new(
            self.inner.opts.context as u64,
            self.inner.opts.concurrency as u64,
        );
        w.prompt_tokens = w.prompt_tokens.min(512);
        w
    }

    async fn planner_loop(self) {
        loop {
            self.inner.replan.notified().await;
            // Let a burst of joins/leaves settle.
            tokio::time::sleep(Duration::from_millis(700)).await;
            if let Err(e) = self.replan_once().await {
                self.event("error", format!("{e:#}"));
                self.set_status(Status::Failed {
                    error: format!("{e:#}"),
                });
            }
        }
    }

    async fn replan_once(&self) -> Result<()> {
        let (cluster, ids) = self.cluster_snapshot();
        if cluster.nodes.is_empty() {
            self.set_status(Status::Waiting {
                reason: "No machines yet. Join one with the command above.".into(),
                advice: vec![],
            });
            return Ok(());
        }
        if cluster.nodes.len() < self.inner.opts.min_stages {
            let reason = format!(
                "waiting for {} machine(s): --min-machines {} asks for a split across at least that many",
                self.inner.opts.min_stages - cluster.nodes.len(),
                self.inner.opts.min_stages
            );
            if !matches!(self.status(), Status::Waiting { .. })
                || self.inner.pipeline.read().await.is_some()
            {
                self.event("info", format!("Waiting: {reason}"));
            }
            if self.inner.pipeline.read().await.is_none() {
                self.set_status(Status::Waiting {
                    reason,
                    advice: vec![],
                });
            }
            return Ok(());
        }
        let opts = PlanOptions {
            goal: self.inner.opts.goal,
            safety_frac: self.inner.opts.safety,
            min_stages: self.inner.opts.min_stages.max(1),
            ..Default::default()
        };
        let w = self.workload();
        let result = planner::plan(&self.inner.spec, &cluster, &w, &opts);
        *self.inner.last_plan.lock().unwrap() = Some(result.clone());
        let Some(best) = result.selected.clone() else {
            let advice = tendril_core::advice::advise(
                &self.inner.spec,
                &cluster,
                &w,
                &opts,
                &result,
                "tendril serve",
            );
            let reason = result
                .rejected
                .first()
                .map(|r| r.reason.clone())
                .unwrap_or_else(|| "no feasible placement".into());
            let adv: Vec<String> = advice
                .iter()
                .map(|a| format!("{} — {}", a.title, a.detail))
                .collect();
            let was_waiting = matches!(self.status(), Status::Waiting { .. });
            if !was_waiting {
                self.event(
                    "warn",
                    format!(
                        "{} doesn't fit on the {} machine(s) here yet: {reason}. Waiting for more machines to join.",
                        self.inner.opts.model_name,
                        cluster.nodes.len()
                    ),
                );
            }
            if self.inner.pipeline.read().await.is_none() {
                self.set_status(Status::Waiting {
                    reason,
                    advice: adv,
                });
            }
            return Ok(());
        };
        // Keep a working pipeline unless the new plan is much better and idle.
        if let Some(cur) = self.inner.pipeline.read().await.as_ref() {
            let same = cur.slots.len() == best.stages.len()
                && cur.slots.iter().zip(&best.stages).all(|(s, b)| {
                    s.node_id == ids[b.node]
                        && s.spec.layer_start as u64 == b.layer_start
                        && s.spec.layer_end as u64 == b.layer_end
                });
            if same {
                return Ok(());
            }
            let gain = best.tokens_per_sec / cur.plan.tokens_per_sec.max(1e-9);
            let busy = !self.inner.router.lock().unwrap().is_empty();
            if gain < 1.25 || busy {
                if gain >= 1.25 {
                    self.event(
                        "info",
                        format!(
                            "A faster plan is available ({} at ~{:.1} tok/s); switching when idle",
                            best.label(),
                            best.tokens_per_sec
                        ),
                    );
                    let c = self.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        c.inner.replan.notify_one();
                    });
                }
                return Ok(());
            }
            self.event(
                "info",
                format!(
                    "Switching to a faster plan: {} (~{:.1} tok/s)",
                    best.label(),
                    best.tokens_per_sec
                ),
            );
            self.teardown("switching plans").await;
        }
        self.activate(best, &ids).await
    }

    // ------------------------------------------------------------------
    // Loading

    fn stage_status(&self, plan: &Plan, phase: &str) -> Vec<StageStatus> {
        plan.stages
            .iter()
            .map(|s| StageStatus {
                node: s.node_name.clone(),
                components: s.describe_components(),
                layers: (s.layer_start, s.layer_end),
                phase: phase.into(),
                done: 0,
                total: 0,
                device: s.backend.label().into(),
            })
            .collect()
    }

    fn update_stage(&self, idx: usize, f: impl FnOnce(&mut StageStatus)) {
        let mut st = self.inner.status.lock().unwrap();
        if let Status::Loading { stages, .. } = &mut *st {
            if let Some(s) = stages.get_mut(idx) {
                f(s);
            }
        }
    }

    async fn activate(&self, plan: Plan, ids: &[u32]) -> Result<()> {
        let epoch = self.inner.epoch.fetch_add(1, Ordering::Relaxed) + 1;
        let label = plan.label();
        self.event(
            "info",
            format!(
                "Plan: {} — ~{:.1} tok/s predicted, first token ~{}",
                describe_plan(&plan),
                plan.tokens_per_sec,
                fmt_ms(plan.ttft_ms)
            ),
        );
        self.set_status(Status::Loading {
            plan: label.clone(),
            stages: self.stage_status(&plan, "waiting"),
        });
        let t0 = Instant::now();
        let slots: Vec<StageSlot> = plan
            .stages
            .iter()
            .map(|s| StageSlot {
                node_id: ids[s.node],
                spec: StageSpec {
                    layer_start: s.layer_start as usize,
                    layer_end: s.layer_end as usize,
                    embed: s.embed,
                    head: s.head,
                },
            })
            .collect();
        let n = slots.len();
        let data_port = self.inner.data_port.load(Ordering::Relaxed) as u16;

        // Addresses: where stage i listens, and how stage i reaches the coordinator.
        let remotes: Vec<Option<Arc<RemoteNode>>> =
            slots.iter().map(|s| self.remote(s.node_id)).collect();
        if slots
            .iter()
            .zip(&remotes)
            .any(|(s, r)| r.is_none() && !self.is_local(s.node_id))
        {
            bail!("a planned machine disappeared before loading");
        }
        let stage_addr = |i: usize, from: Option<&Arc<RemoteNode>>| -> String {
            match &remotes[i] {
                Some(r) => format!("{}:{}", r.peer_ip, r.data_port),
                // The local stage listens on the coordinator's data port.
                None => match from {
                    Some(f) => format!("{}:{}", f.local_ip, data_port),
                    None => format!("127.0.0.1:{data_port}"),
                },
            }
        };
        let (results_tx, results_rx) = mpsc::unbounded_channel::<Msg>();
        tokio::spawn(self.clone().route_results(results_rx));

        let mut local_worker: Option<Arc<StageWorker>> = None;
        let mut local_next: Option<(NextHop, u64)> = None;
        let mut loads = Vec::new();
        for i in 0..n {
            let next = if i + 1 < n {
                NextHop::Stage(stage_addr(i + 1, remotes[i].as_ref()))
            } else {
                match &remotes[i] {
                    Some(r) => NextHop::Coordinator(format!("{}:{}", r.local_ip, data_port)),
                    None => NextHop::Coordinator("local".into()),
                }
            };
            let spec = slots[i].spec;
            match &remotes[i] {
                Some(r) => {
                    let tensors = stage_tensors(&self.inner.ws, &self.inner.cfg, &spec)?;
                    loads.push(tokio::spawn(self.clone().load_remote(
                        i,
                        epoch,
                        r.clone(),
                        spec,
                        tensors,
                        next,
                    )));
                }
                None => {
                    self.update_stage(i, |s| s.phase = "loading".into());
                    let ws = self.inner.ws.clone();
                    let cfg = self.inner.cfg.clone();
                    let device = self.inner.opts.device.clone();
                    let fmt = self.inner.opts.format;
                    let stage = tokio::task::spawn_blocking(move || -> Result<Stage> {
                        let dev = device_from_name(&device)?;
                        let dtype = activation_dtype(&dev, &cfg.torch_dtype);
                        Stage::load(
                            cfg,
                            &ws,
                            spec,
                            &LoadOptions {
                                format: fmt,
                                device: dev,
                                dtype,
                            },
                        )
                    })
                    .await??;
                    let dev = device_label(&stage.device).to_string();
                    let bytes = stage.weight_bytes as u64;
                    self.update_stage(i, |s| {
                        s.phase = "ready".into();
                        s.done = bytes;
                        s.total = bytes;
                        s.device = dev.clone();
                    });
                    self.event(
                        "ok",
                        format!(
                            "{} loaded {} ({}, {})",
                            self.node_name(slots[i].node_id),
                            spec_label(&spec),
                            Bytes(bytes),
                            dev
                        ),
                    );
                    // Output goes to the next stage or straight to the router.
                    let (tx, rx) = mpsc::unbounded_channel::<Msg>();
                    let w = Arc::new(StageWorker::spawn(stage, i as u32, epoch, tx));
                    local_next = Some((next.clone(), epoch));
                    match next {
                        NextHop::Coordinator(_) => {
                            let rt = results_tx.clone();
                            tokio::spawn(async move {
                                let mut rx = rx;
                                while let Some(m) = rx.recv().await {
                                    if rt.send(m).is_err() {
                                        break;
                                    }
                                }
                            });
                        }
                        NextHop::Stage(addr) => {
                            let psk = self.inner.psk;
                            tokio::spawn(async move {
                                match connect_retry(&addr, &psk, Duration::from_secs(600)).await {
                                    Ok(conn) => pump(conn, epoch, false, rx).await,
                                    Err(e) => {
                                        tracing::warn!("local stage cannot reach next stage: {e:#}")
                                    }
                                }
                            });
                        }
                    }
                    local_worker = Some(w);
                }
            }
        }
        let _ = local_next;
        *self.inner.local_worker.lock().unwrap() = local_worker.clone().map(|w| (epoch, w));
        for l in loads {
            match l.await? {
                Ok(()) => {}
                Err(e) => {
                    self.teardown("load failed").await;
                    self.set_status(Status::Failed {
                        error: format!("{e:#}"),
                    });
                    return Err(e);
                }
            }
        }
        // The entry point: the first stage.
        let entry = match &remotes[0] {
            None => Entry::Local(local_worker.clone().expect("local first stage")),
            Some(r) => {
                let conn = connect_retry(
                    &format!("{}:{}", r.peer_ip, r.data_port),
                    &self.inner.psk,
                    Duration::from_secs(30),
                )
                .await?;
                let (tx, rx) = mpsc::unbounded_channel();
                tokio::spawn(pump(conn, epoch, false, rx));
                Entry::Remote(tx)
            }
        };
        let predicted = plan.tokens_per_sec;
        let stages = {
            let st = self.inner.status.lock().unwrap();
            match &*st {
                Status::Loading { stages, .. } => stages.clone(),
                _ => self.stage_status(&plan, "ready"),
            }
        };
        let pipeline = Pipeline {
            epoch,
            plan,
            slots,
            entry,
            permits: Arc::new(Semaphore::new(self.inner.opts.concurrency.max(1))),
            _local: local_worker,
            ready_at: Instant::now(),
            telemetry: StdMutex::new(Telemetry::default()),
        };
        *self.inner.pipeline.write().await = Some(Arc::new(pipeline));
        self.set_status(Status::Ready {
            plan: label,
            stages,
            predicted_tps: predicted,
            since_ms: 0,
        });
        self.event(
            "ok",
            format!("Ready in {:.1} s", t0.elapsed().as_secs_f64()),
        );
        Ok(())
    }

    fn is_local(&self, id: u32) -> bool {
        self.inner
            .nodes
            .lock()
            .unwrap()
            .iter()
            .any(|n| n.id == id && n.remote.is_none())
    }

    fn node_name(&self, id: u32) -> String {
        self.inner
            .nodes
            .lock()
            .unwrap()
            .iter()
            .find(|n| n.id == id)
            .map(|n| n.profile.name.clone())
            .unwrap_or_else(|| format!("node{id}"))
    }

    async fn load_remote(
        self,
        idx: usize,
        epoch: u64,
        r: Arc<RemoteNode>,
        spec: StageSpec,
        tensors: Vec<TensorEntry>,
        next: NextHop,
    ) -> Result<()> {
        let (itx, mut irx) = mpsc::unbounded_channel();
        *r.inbox.lock().unwrap() = Some(itx);
        let name = self
            .inner
            .nodes
            .lock()
            .unwrap()
            .iter()
            .find(|n| n.remote.as_ref().is_some_and(|x| Arc::ptr_eq(x, &r)))
            .map(|n| n.profile.name.clone())
            .unwrap_or_default();
        r.tx.send(Msg::LoadStage {
            epoch,
            index: idx as u32,
            model_key: self.inner.model_key.clone(),
            config_json: self.inner.config_json.clone(),
            spec,
            format: self.inner.opts.format.label().to_string(),
            device: "auto".into(),
            tensors: tensors.clone(),
            next,
        })
        .await
        .map_err(|_| anyhow!("{name} disconnected"))?;
        let total: u64 = tensors.iter().map(|t| t.len).sum();
        self.update_stage(idx, |s| {
            s.phase = "preparing".into();
            s.total = total;
        });
        loop {
            let m = tokio::time::timeout(Duration::from_secs(600), irx.recv())
                .await
                .map_err(|_| anyhow!("{name} did not finish loading within 10 minutes"))?
                .ok_or_else(|| anyhow!("{name} disconnected while loading"))?;
            match m {
                Msg::NeedWeights { epoch: e, names } if e == epoch => {
                    if names.is_empty() {
                        self.event(
                            "info",
                            format!(
                                "{name} already has {} cached — skipping transfer",
                                spec_label(&spec)
                            ),
                        );
                        continue;
                    }
                    self.event(
                        "info",
                        format!(
                            "Sending {} of weights to {name} ({})",
                            Bytes(total),
                            spec_label(&spec)
                        ),
                    );
                    self.update_stage(idx, |s| s.phase = "sending".into());
                    let mut sent = 0u64;
                    let t0 = Instant::now();
                    const CHUNK: usize = 4 << 20;
                    for nm in names {
                        let (_, _, data) = self.inner.ws.raw(&nm)?;
                        for (k, part) in data.chunks(CHUNK).enumerate() {
                            r.tx.send(Msg::WeightData {
                                epoch,
                                name: nm.clone(),
                                offset: (k * CHUNK) as u64,
                                data: part.to_vec(),
                            })
                            .await
                            .map_err(|_| anyhow!("{name} disconnected during transfer"))?;
                            sent += part.len() as u64;
                            self.update_stage(idx, |s| s.done = sent);
                        }
                    }
                    r.tx.send(Msg::WeightsDone { epoch })
                        .await
                        .map_err(|_| anyhow!("{name} disconnected"))?;
                    let secs = t0.elapsed().as_secs_f64();
                    self.event(
                        "info",
                        format!(
                            "Sent {} to {name} in {:.1} s ({:.2} Gb/s)",
                            Bytes(sent),
                            secs,
                            sent as f64 * 8.0 / secs.max(1e-3) / 1e9
                        ),
                    );
                }
                Msg::LoadProgress {
                    epoch: e, phase, ..
                } if e == epoch => {
                    self.update_stage(idx, |s| s.phase = phase);
                }
                Msg::StageReady {
                    epoch: e,
                    weight_bytes,
                    load_ms,
                    device,
                } if e == epoch => {
                    self.update_stage(idx, |s| {
                        s.phase = "ready".into();
                        s.done = s.total;
                        s.device = device.clone();
                    });
                    self.event(
                        "ok",
                        format!(
                            "{name} loaded {} ({}, {}) in {:.1} s",
                            spec_label(&spec),
                            Bytes(weight_bytes),
                            device,
                            load_ms as f64 / 1000.0
                        ),
                    );
                    *r.inbox.lock().unwrap() = None;
                    return Ok(());
                }
                Msg::LoadFailed { epoch: e, error } if e == epoch => {
                    *r.inbox.lock().unwrap() = None;
                    bail!("{name} could not load its stage: {error}");
                }
                _ => {}
            }
        }
    }

    // ------------------------------------------------------------------
    // Data plane

    async fn data_loop(self, listener: TcpListener) {
        loop {
            let Ok((s, _)) = listener.accept().await else {
                continue;
            };
            let c = self.clone();
            tokio::spawn(async move {
                let Ok(mut conn) = wire::accept(s, &c.inner.psk).await else {
                    return;
                };
                let Ok(Msg::DataHello { epoch, results }) = conn.reader.recv().await else {
                    return;
                };
                while let Ok(m) = conn.reader.recv().await {
                    if results {
                        c.route(m);
                    } else {
                        // Upstream input for this machine's stage.
                        let w = c.inner.local_worker.lock().unwrap().clone();
                        if let Some((e, w)) = w {
                            if e == epoch {
                                w.submit(m);
                            }
                        }
                    }
                }
            });
        }
    }

    async fn route_results(self, mut rx: mpsc::UnboundedReceiver<Msg>) {
        while let Some(m) = rx.recv().await {
            self.route(m);
        }
    }

    fn route(&self, m: Msg) {
        let (seq, ev) = match m {
            Msg::Token {
                seq, token, trace, ..
            } => (seq, SeqEvent::Token(token, trace)),
            Msg::StageError {
                seq, stage, error, ..
            } => (
                seq,
                SeqEvent::Error(format!("stage {} failed: {error}", stage + 1)),
            ),
            _ => return,
        };
        if let Some(tx) = self.inner.router.lock().unwrap().get(&seq) {
            let _ = tx.send(ev);
        }
    }

    /// Start generating. Text arrives on the returned channel.
    pub async fn generate(&self, req: GenRequest) -> Result<mpsc::Receiver<GenOut>, ServeError> {
        let queued_at = Instant::now();
        let pipeline = self.inner.pipeline.read().await.clone();
        let Some(p) = pipeline else {
            let why = match self.status() {
                Status::Waiting { reason, .. } => format!("the model is not running yet: {reason}"),
                Status::Loading { .. } => {
                    "the model is still loading; try again in a moment".into()
                }
                Status::Failed { error } => format!("the cluster failed: {error}"),
                _ => "the model is not ready yet".into(),
            };
            return Err(ServeError::NotReady(why));
        };
        let ctx = self.inner.opts.context;
        if req.prompt.is_empty() {
            return Err(ServeError::BadRequest("the prompt is empty".into()));
        }
        if req.prompt.len() >= ctx {
            return Err(ServeError::BadRequest(format!(
                "the prompt is {} tokens but this server was started with a {} token context (restart with a larger --context if memory allows)",
                req.prompt.len(),
                ctx
            )));
        }
        let max_tokens = req.max_tokens.min(ctx - req.prompt.len()).max(1);
        {
            let mut m = self.inner.metrics.lock().unwrap();
            m.queued += 1;
        }
        let permit =
            tokio::time::timeout(Duration::from_secs(120), p.permits.clone().acquire_owned()).await;
        {
            let mut m = self.inner.metrics.lock().unwrap();
            m.queued -= 1;
        }
        let permit = match permit {
            Ok(Ok(p)) => p,
            _ => {
                return Err(ServeError::Busy(format!(
                    "all {} generation slots are busy; try again shortly",
                    self.inner.opts.concurrency
                )))
            }
        };
        let (tx, rx) = mpsc::channel(64);
        let seq = self.inner.next_seq.fetch_add(1, Ordering::Relaxed);
        let (stx, srx) = mpsc::unbounded_channel();
        self.inner.router.lock().unwrap().insert(seq, stx);
        let c = self.clone();
        tokio::spawn(async move {
            let queue_ms = queued_at.elapsed().as_secs_f64() * 1000.0;
            c.drive(
                p,
                permit,
                seq,
                GenRequest { max_tokens, ..req },
                srx,
                tx,
                queue_ms,
            )
            .await;
        });
        Ok(rx)
    }

    #[allow(clippy::too_many_arguments)]
    async fn drive(
        &self,
        p: Arc<Pipeline>,
        _permit: OwnedSemaphorePermit,
        seq: u64,
        req: GenRequest,
        mut srx: mpsc::UnboundedReceiver<SeqEvent>,
        out: mpsc::Sender<GenOut>,
        queue_ms: f64,
    ) {
        {
            let mut m = self.inner.metrics.lock().unwrap();
            m.active += 1;
            m.requests += 1;
        }
        let t0 = Instant::now();
        let mut timing = Timing {
            prompt_tokens: req.prompt.len(),
            queue_ms,
            ..Default::default()
        };
        let epoch = p.epoch;
        let mut pos = 0usize;
        let setup = SampleSetup {
            params: req.params.clone(),
            history: req.prompt.clone(),
        };
        // Prefill in chunks; they pipeline through the stages back to back.
        let mut first = true;
        while pos < req.prompt.len() {
            let nchunk = prefill_chunk(&self.inner.cfg, pos).min(req.prompt.len() - pos);
            let last = pos + nchunk == req.prompt.len();
            p.entry.send(Msg::Forward {
                epoch,
                seq,
                pos: pos as u32,
                payload: Payload::Tokens(req.prompt[pos..pos + nchunk].to_vec()),
                want_logits: last,
                sample: if first { Some(setup.clone()) } else { None },
                trace: Vec::new(),
            });
            first = false;
            pos += nchunk;
        }
        let mut detok = Detokenizer::new();
        let mut stop = StopMatcher::new(req.stop.clone());
        let mut reason = FinishReason::Length;
        let mut error: Option<String> = None;
        let mut step_sent: Option<Instant> = None;
        loop {
            let ev = match tokio::time::timeout(Duration::from_secs(300), srx.recv()).await {
                Ok(Some(e)) => e,
                Ok(None) => {
                    error = Some("the pipeline stopped".into());
                    break;
                }
                Err(_) => {
                    error = Some("no token for 5 minutes; the pipeline seems stuck".into());
                    break;
                }
            };
            let tok = match ev {
                SeqEvent::Token(t, trace) => {
                    if let Some(sent) = step_sent.take().filter(|_| pos <= 2048) {
                        // Calibrate on typical contexts only; very long ones are attention-bound.
                        let step_ms = sent.elapsed().as_secs_f64() * 1000.0;
                        let samples = {
                            let mut tel = p.telemetry.lock().unwrap();
                            tel.record(step_ms, &trace, p.slots.len());
                            tel.samples
                        };
                        if samples == 64 || samples % 512 == 0 {
                            self.calibrate(&p);
                        }
                    }
                    t
                }
                SeqEvent::Error(e) => {
                    error = Some(e);
                    break;
                }
            };
            if timing.completion_tokens == 0 {
                timing.ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            timing.completion_tokens += 1;
            if self.inner.tok.is_stop(tok) && !req.ignore_eos {
                reason = FinishReason::Stop;
                break;
            }
            let delta = detok.push(&self.inner.tok, tok).unwrap_or_default();
            let (emit, stopped) = stop.push(&delta);
            if !emit.is_empty() && out.send(GenOut::Text(emit)).await.is_err() {
                reason = FinishReason::Cancelled;
                break;
            }
            if stopped {
                reason = FinishReason::Stop;
                break;
            }
            if timing.completion_tokens >= req.max_tokens {
                break;
            }
            p.entry.send(Msg::Forward {
                epoch,
                seq,
                pos: pos as u32,
                payload: Payload::Tokens(vec![tok]),
                want_logits: true,
                sample: None,
                trace: Vec::new(),
            });
            step_sent = Some(Instant::now());
            pos += 1;
        }
        let rest = stop.flush();
        if !rest.is_empty() && reason != FinishReason::Stop && error.is_none() {
            let _ = out.send(GenOut::Text(rest)).await;
        }
        // Free KV along the whole chain.
        p.entry.send(Msg::Release { epoch, seq });
        self.inner.router.lock().unwrap().remove(&seq);
        timing.total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        {
            let mut m = self.inner.metrics.lock().unwrap();
            m.active -= 1;
            m.prompt_tokens += timing.prompt_tokens as u64;
            m.completion_tokens += timing.completion_tokens as u64;
            if error.is_some() {
                m.errors += 1;
            }
            m.recent.push_back(RequestRecord {
                at: now(),
                prompt_tokens: timing.prompt_tokens,
                completion_tokens: timing.completion_tokens,
                ttft_ms: timing.ttft_ms,
                decode_tps: timing.decode_tps(),
                total_ms: timing.total_ms,
                finish: match (&error, reason) {
                    (Some(_), _) => "error".into(),
                    (None, r) => r.openai().into(),
                },
            });
            while m.recent.len() > 50 {
                m.recent.pop_front();
            }
        }
        match error {
            Some(e) => {
                let _ = out.send(GenOut::Error(e)).await;
            }
            None => {
                let _ = out.send(GenOut::Done { reason, timing }).await;
            }
        }
    }

    /// Compare measured stage speed with the plan's prediction and remember
    /// the difference, so future plans use how fast machines really are.
    fn calibrate(&self, p: &Pipeline) {
        let tel = p.telemetry.lock().unwrap().clone();
        let names: Vec<(String, String)> = {
            let nodes = self.inner.nodes.lock().unwrap();
            p.slots
                .iter()
                .map(|s| {
                    nodes
                        .iter()
                        .find(|n| n.id == s.node_id)
                        .map(|n| (n.profile.name.clone(), n.profile.chip.clone()))
                        .unwrap_or_default()
                })
                .collect()
        };
        let mut notes = Vec::new();
        {
            let mut cal = self.inner.calibration.lock().unwrap();
            for (i, st) in p.plan.stages.iter().enumerate() {
                let measured = tel.compute_ms.get(i).copied().unwrap_or(0.0);
                if measured <= 0.0 || st.decode_ms <= 0.0 {
                    continue;
                }
                let (name, chip) = &names[i];
                if let Some(f) = cal.observe(name, chip, measured / st.decode_ms, tel.samples) {
                    notes.push(format!(
                        "{name} computes its part in {} vs {} predicted ({f:.2}× the spec-sheet estimate)",
                        fmt_ms(measured),
                        fmt_ms(st.decode_ms)
                    ));
                }
            }
            cal.save();
        }
        for n in notes {
            self.event(
                "info",
                format!("Calibrated: {n}; future plans use the measured speed"),
            );
        }
    }

    /// Measured timings of the running pipeline.
    pub async fn telemetry(&self) -> Option<(Plan, Telemetry)> {
        self.inner
            .pipeline
            .read()
            .await
            .as_ref()
            .map(|p| (p.plan.clone(), p.telemetry.lock().unwrap().clone()))
    }

    /// Seconds since the pipeline became ready.
    pub async fn ready_for(&self) -> Option<Duration> {
        self.inner
            .pipeline
            .read()
            .await
            .as_ref()
            .map(|p| p.ready_at.elapsed())
    }

    pub async fn shutdown(&self) {
        let remotes: Vec<Arc<RemoteNode>> = self
            .inner
            .nodes
            .lock()
            .unwrap()
            .iter()
            .filter_map(|n| n.remote.clone())
            .collect();
        self.teardown("server stopped").await;
        for r in remotes {
            let _ = r.tx.try_send(Msg::Bye {
                reason: "coordinator shutting down".into(),
            });
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Respect a user's `--max-memory` cap.
pub fn apply_memory_cap(p: &mut NodeProfile, cap: Option<Bytes>) {
    if let Some(cap) = cap {
        if cap < p.usable_memory {
            p.usable_memory = cap;
            p.usable_reason = format!("limited to {cap} by --max-memory");
        }
    }
}

pub fn spec_label(s: &StageSpec) -> String {
    let mut parts = Vec::new();
    if s.embed {
        parts.push("embeddings".to_string());
    }
    if s.layer_end > s.layer_start {
        parts.push(format!("layers {}–{}", s.layer_start, s.layer_end - 1));
    }
    if s.head {
        parts.push("head".to_string());
    }
    parts.join(" + ")
}

pub fn describe_plan(p: &Plan) -> String {
    p.stages
        .iter()
        .map(|s| format!("{} [{}]", s.node_name, s.describe_components()))
        .collect::<Vec<_>>()
        .join(" → ")
}
