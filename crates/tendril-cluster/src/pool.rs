//! The server's view of the cluster: machines, the engines on them, and the
//! models deployed across them.
//!
//! Loading a model is: pick an engine, resolve the model with it, plan where
//! it goes with Tendril's planner (on memory the other models left free), let
//! the engine adjust that to how it really lays the model out, then start
//! the engine's processes on the chosen machines in order and expose its API
//! through an encrypted tunnel. The plan shown before loading is computed by
//! the same code, so it is the plan that runs.

use crate::machine::clock;
use crate::proto::{Msg, Peer, RoleState, PROTOCOL};
use crate::tunnel::{Forwarder, Route};
use crate::{token, wire};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tendril_core::cluster::{Cluster, Link};
use tendril_core::hardware::NodeProfile;
use tendril_core::planner::{plan, PlanOptions, Workload};
use tendril_core::Bytes;
use tendril_engines::{
    Detection, Engine, EngineInfo, Machine, Missing, ModelRequest, MultiMachine, Placement,
    ResolvedModel, Role, Security, Stage, Task,
};
use tokio::net::TcpListener;
use tokio::sync::{broadcast, mpsc, Notify};

/// What to load.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LoadRequest {
    /// Engine id ("llama.cpp", "vllm", "ollama", …).
    pub engine: String,
    pub model: String,
    #[serde(default)]
    pub quantize: Option<String>,
    /// Tokens per conversation.
    #[serde(default)]
    pub context: Option<usize>,
    /// Conversations at once.
    #[serde(default)]
    pub concurrency: Option<usize>,
    /// Use only these machines (keys or names).
    #[serde(default)]
    pub machines: Option<Vec<String>>,
    /// Split across at least this many machines (engines that split layers).
    #[serde(default)]
    pub min_machines: Option<usize>,
    /// The user confirmed the machines share a trusted network (needed when
    /// an engine's own traffic between machines can't be encrypted).
    #[serde(default)]
    pub trusted_network: bool,
    /// Load only if the pool is still as it was when this plan was shown.
    #[serde(default)]
    pub expect_version: Option<u64>,
}

/// The pool changed between showing a plan and confirming it.
#[derive(Debug)]
pub struct PoolChanged;

impl std::fmt::Display for PoolChanged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the machines or models changed since this plan was made — plan again to see the new split")
    }
}

impl std::error::Error for PoolChanged {}

/// Loading needs the user's confirmation that the network is trusted.
#[derive(Debug)]
pub struct NeedsTrust(pub String);

impl std::fmt::Display for NeedsTrust {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NeedsTrust {}

#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub at: String,
    pub level: &'static str,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

struct MachineEntry {
    key: String,
    profile: NodeProfile,
    engines: Vec<Detection>,
    tx: mpsc::UnboundedSender<Msg>,
    address: String,
    tunnel_port: u16,
    local: bool,
    rtt_ms: Option<f64>,
    session: u64,
}

impl MachineEntry {
    fn detection(&self, engine: &str) -> Detection {
        self.engines
            .iter()
            .find(|d| d.engine == engine)
            .cloned()
            .unwrap_or_else(|| Detection {
                engine: engine.into(),
                ..Default::default()
            })
    }

    fn tunnel(&self) -> String {
        format!("{}:{}", self.address, self.tunnel_port)
    }
}

/// One machine, for the dashboard.
#[derive(Clone, Debug, Serialize)]
pub struct MachineView {
    pub key: String,
    pub name: String,
    pub chip: String,
    pub backend: String,
    pub os: String,
    pub cpu_cores: u32,
    pub gpu_cores: Option<u32>,
    pub total_memory: u64,
    pub usable_memory: u64,
    /// Not promised to any model.
    pub free_memory: u64,
    pub bandwidth_gbs: f64,
    pub tflops: f64,
    pub measured: bool,
    pub local: bool,
    pub address: String,
    pub rtt_ms: Option<f64>,
    pub on_battery: Option<bool>,
    pub engines: Vec<Detection>,
    /// What runs here: (model, role, state).
    pub roles: Vec<(String, String, String)>,
}

/// One machine's part of a planned model.
#[derive(Clone, Debug, Serialize)]
pub struct StageView {
    pub machine: String,
    pub key: String,
    pub backend: String,
    pub layers: (u64, u64),
    pub weight_bytes: u64,
    pub weight_pct: f64,
    /// Peak memory it needs and what is free there.
    pub memory_needed: u64,
    pub memory_free: u64,
    pub serves_api: bool,
    /// What it downloads or receives.
    pub transfer: String,
}

/// A machine that can't run the chosen engine, and why.
#[derive(Clone, Debug, Serialize)]
pub struct Unavailable {
    pub machine: String,
    pub missing: Vec<Missing>,
    pub notes: Vec<String>,
}

/// What loading a model would do, before anything is downloaded.
#[derive(Clone, Debug, Serialize)]
pub struct PlanPreview {
    pub pool_version: u64,
    pub engine: EngineInfo,
    pub model: String,
    pub task: Task,
    pub vision: bool,
    pub weight_bytes: u64,
    pub layers: u64,
    pub layered: bool,
    pub context: usize,
    pub concurrency: usize,
    pub fits: bool,
    pub reason: Option<String>,
    pub stages: Vec<StageView>,
    pub tokens_per_sec: Option<f64>,
    pub ttft_ms: Option<f64>,
    /// The model spans machines and the engine's own traffic can't be
    /// encrypted: loading needs `trusted_network`.
    pub needs_trusted_network: bool,
    /// How traffic between the machines is protected for this plan.
    pub security: String,
    pub risk: Option<String>,
    pub unavailable: Vec<Unavailable>,
    pub notes: Vec<String>,
    /// The processes it starts: (machine, command).
    pub commands: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Loading,
    Ready,
    Failed,
    Stopping,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoleView {
    pub id: String,
    pub machine: String,
    pub machine_name: String,
    pub serves_api: bool,
    /// "waiting", "downloading", "starting", "loading", "preparing", "ready", "failed", "exited".
    pub state: String,
    pub detail: String,
    pub done: u64,
    pub total: u64,
    #[serde(skip)]
    pub sent: bool,
    #[serde(skip)]
    pub ports: Option<Vec<(String, u16)>>,
    #[serde(skip)]
    pub ready: bool,
    pub log: Vec<String>,
}

struct DepState {
    status: Status,
    reason: Option<String>,
    roles: Vec<RoleView>,
    api: Option<Forwarder>,
}

/// A model running (or loading) with one engine on some machines.
pub struct Deployment {
    pub id: String,
    /// The name clients use (`"model"` in API requests).
    pub name: String,
    pub engine: Arc<dyn Engine>,
    pub model: ResolvedModel,
    pub roles: Vec<Role>,
    /// Memory promised per machine key.
    pub budgets: Vec<(String, u64)>,
    pub preview: PlanPreview,
    pub started: Instant,
    state: Mutex<DepState>,
    changed: Notify,
    pub requests: AtomicU64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeploymentView {
    pub id: String,
    pub name: String,
    pub engine: String,
    pub engine_name: String,
    pub task: Task,
    pub vision: bool,
    pub status: Status,
    pub reason: Option<String>,
    pub roles: Vec<RoleView>,
    pub plan: PlanPreview,
    pub uptime_s: u64,
    pub requests: u64,
}

impl Deployment {
    pub fn status(&self) -> Status {
        self.state.lock().unwrap().status
    }

    /// Local port of the engine's API, once it is ready.
    pub fn api_port(&self) -> Option<u16> {
        let s = self.state.lock().unwrap();
        if s.status != Status::Ready {
            return None;
        }
        s.api.as_ref().map(|f| f.port)
    }

    pub fn view(&self) -> DeploymentView {
        let s = self.state.lock().unwrap();
        DeploymentView {
            id: self.id.clone(),
            name: self.name.clone(),
            engine: self.engine.info().id,
            engine_name: self.engine.info().name,
            task: self.model.task,
            vision: self.model.vision,
            status: s.status,
            reason: s.reason.clone(),
            roles: s.roles.clone(),
            plan: self.preview.clone(),
            uptime_s: self.started.elapsed().as_secs(),
            requests: self.requests.load(Ordering::Relaxed),
        }
    }
}

pub struct PoolOptions {
    pub control_port: u16,
    pub token: String,
    /// Shown to joining machines.
    pub name: String,
    pub default_context: usize,
    pub default_concurrency: usize,
}

pub struct Pool {
    opts: PoolOptions,
    psk: [u8; 32],
    machines: Mutex<BTreeMap<String, MachineEntry>>,
    deployments: Mutex<Vec<Arc<Deployment>>>,
    events: broadcast::Sender<Event>,
    history: Mutex<VecDeque<Event>>,
    version: AtomicU64,
    resolved: Mutex<HashMap<String, (Instant, ResolvedModel)>>,
    next_id: AtomicU64,
    sessions: AtomicU64,
    stop: Notify,
    stopping: AtomicBool,
    changed: Notify,
    pub control_port: u16,
}

/// Memory a stage may use on top of its planned peak (rounding, engine slack).
fn budget(peak: u64) -> u64 {
    (peak as f64 * 1.05) as u64 + (64 << 20)
}

type Planned = (
    PlanPreview,
    ResolvedModel,
    Vec<Role>,
    Vec<(String, u64)>,
    Arc<dyn Engine>,
);

impl Pool {
    /// Start listening for machines.
    pub async fn start(opts: PoolOptions) -> Result<Arc<Pool>> {
        let listener = TcpListener::bind(("0.0.0.0", opts.control_port))
            .await
            .with_context(|| format!("cannot listen on control port {}", opts.control_port))?;
        let control_port = listener.local_addr()?.port();
        let (events, _) = broadcast::channel(256);
        let pool = Arc::new(Pool {
            psk: token::psk(&opts.token),
            opts,
            machines: Mutex::new(BTreeMap::new()),
            deployments: Mutex::new(Vec::new()),
            events,
            history: Mutex::new(VecDeque::new()),
            version: AtomicU64::new(1),
            resolved: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            sessions: AtomicU64::new(1),
            stop: Notify::new(),
            stopping: AtomicBool::new(false),
            changed: Notify::new(),
            control_port,
        });
        tokio::spawn(pool.clone().accept_loop(listener));
        tokio::spawn(pool.clone().ping_loop());
        Ok(pool)
    }

    pub fn token(&self) -> &str {
        &self.opts.token
    }

    pub fn name(&self) -> &str {
        &self.opts.name
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    fn bump(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
        self.changed.notify_waiters();
    }

    pub fn event(&self, level: &'static str, text: impl Into<String>, model: Option<&str>) {
        let e = Event {
            at: clock(),
            level,
            text: text.into(),
            model: model.map(String::from),
        };
        {
            let mut h = self.history.lock().unwrap();
            h.push_back(e.clone());
            while h.len() > 200 {
                h.pop_front();
            }
        }
        let _ = self.events.send(e);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.events.subscribe()
    }

    pub fn history(&self) -> Vec<Event> {
        self.history.lock().unwrap().iter().cloned().collect()
    }

    // ---- machines -------------------------------------------------------

    async fn accept_loop(self: Arc<Self>, listener: TcpListener) {
        loop {
            let Ok((s, _)) = listener.accept().await else {
                continue;
            };
            let me = self.clone();
            tokio::spawn(async move {
                if let Err(e) = me.session(s).await {
                    tracing::debug!("session ended: {e:#}");
                }
            });
        }
    }

    async fn session(self: Arc<Self>, s: tokio::net::TcpStream) -> Result<()> {
        let conn = wire::accept(s, &self.psk).await?;
        let peer = conn.peer;
        let (mut reader, mut writer) = (conn.reader, conn.writer);
        let hello = tokio::time::timeout(Duration::from_secs(30), reader.recv()).await??;
        let Msg::Hello {
            protocol,
            version,
            profile,
            tunnel_port,
            machine,
            local,
            engines,
        } = hello
        else {
            bail!("expected Hello");
        };
        if protocol != PROTOCOL {
            writer
                .send(&Msg::Reject {
                    reason: format!(
                        "this machine runs Tendril {version} (protocol {protocol}); the server runs {} (protocol {PROTOCOL}) — use the same version on every machine",
                        env!("CARGO_PKG_VERSION")
                    ),
                })
                .await?;
            return Ok(());
        }
        if self.stopping.load(Ordering::Relaxed) {
            writer
                .send(&Msg::Reject {
                    reason: "the cluster is stopping".into(),
                })
                .await?;
            return Ok(());
        }
        let key = if local {
            "local".to_string()
        } else {
            format!("{:012x}", machine & 0xffff_ffff_ffff)
        };
        let address = if local {
            "127.0.0.1".to_string()
        } else {
            match peer.ip() {
                std::net::IpAddr::V6(v6) => v6
                    .to_ipv4_mapped()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| v6.to_string()),
                ip => ip.to_string(),
            }
        };
        writer
            .send(&Msg::Welcome {
                key: profile.name.clone(),
                cluster: self.opts.name.clone(),
                address: address.clone(),
            })
            .await?;
        let engines: Vec<Detection> = serde_json::from_str(&engines).unwrap_or_default();
        let (tx, mut rx) = mpsc::unbounded_channel::<Msg>();
        let session = self.sessions.fetch_add(1, Ordering::Relaxed);
        let name = profile.name.clone();
        let ready: Vec<String> = engines
            .iter()
            .filter(|d| d.ready)
            .map(|d| d.engine.clone())
            .collect();
        let replaced = self.machines.lock().unwrap().insert(
            key.clone(),
            MachineEntry {
                key: key.clone(),
                profile,
                engines,
                tx: tx.clone(),
                address,
                tunnel_port,
                local,
                rtt_ms: None,
                session,
            },
        );
        if replaced.is_some() {
            // The machine reconnected: whatever it ran is gone.
            self.machine_lost(&key, &name);
        }
        self.bump();
        self.event(
            "info",
            if local {
                format!(
                    "This machine ({name}) shares its compute · engines: {}",
                    list_or_none(&ready)
                )
            } else {
                format!("{name} joined · engines: {}", list_or_none(&ready))
            },
            None,
        );
        let writer_task = tokio::spawn(async move {
            while let Some(m) = rx.recv().await {
                if writer.send(&m).await.is_err() {
                    break;
                }
            }
        });
        let r: Result<()> = async {
            loop {
                let m = tokio::time::timeout(Duration::from_secs(30), reader.recv())
                    .await
                    .context("machine stopped responding")??;
                match m {
                    Msg::Pong { t } => {
                        let rtt = (now_us().saturating_sub(t)) as f64 / 1000.0;
                        if let Some(e) = self.machines.lock().unwrap().get_mut(&key) {
                            e.rtt_ms = Some(rtt);
                        }
                    }
                    Msg::Engines { engines } => {
                        let d: Vec<Detection> = serde_json::from_str(&engines).unwrap_or_default();
                        let ready: Vec<String> = d
                            .iter()
                            .filter(|x| x.ready)
                            .map(|x| x.engine.clone())
                            .collect();
                        if let Some(e) = self.machines.lock().unwrap().get_mut(&key) {
                            e.engines = d;
                        }
                        self.bump();
                        self.event(
                            "info",
                            format!("{name}: engines ready: {}", list_or_none(&ready)),
                            None,
                        );
                    }
                    Msg::RoleUpdate {
                        deployment,
                        role,
                        state,
                    } => self.role_update(&deployment, &role, state),
                    _ => {}
                }
            }
        }
        .await;
        writer_task.abort();
        let still_ours = {
            let mut ms = self.machines.lock().unwrap();
            let ours = ms.get(&key).is_some_and(|e| e.session == session);
            if ours {
                ms.remove(&key);
            }
            ours
        };
        if still_ours {
            self.machine_lost(&key, &name);
            self.bump();
            if !self.stopping.load(Ordering::Relaxed) {
                let why = r
                    .as_ref()
                    .err()
                    .map(|e| format!(" ({e:#})"))
                    .unwrap_or_default();
                self.event("warn", format!("{name} left{why}"), None);
            }
        }
        r
    }

    async fn ping_loop(self: Arc<Self>) {
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let t = now_us();
            for e in self.machines.lock().unwrap().values() {
                let _ = e.tx.send(Msg::Ping { t });
            }
        }
    }

    /// Deployments that used a machine that went away fail.
    fn machine_lost(&self, key: &str, name: &str) {
        let deps = self.deployments();
        for d in deps {
            if d.roles.iter().any(|r| r.machine == key) {
                self.fail(&d, format!("{name} left the cluster"));
            }
        }
    }

    pub fn machine_views(&self) -> Vec<MachineView> {
        let deps = self.deployments();
        let ms = self.machines.lock().unwrap();
        ms.values()
            .map(|e| {
                let used: u64 = deps
                    .iter()
                    .flat_map(|d| d.budgets.iter())
                    .filter(|(k, _)| *k == e.key)
                    .map(|(_, b)| *b)
                    .sum();
                let mut roles = Vec::new();
                for d in &deps {
                    for r in d
                        .state
                        .lock()
                        .unwrap()
                        .roles
                        .iter()
                        .filter(|r| r.machine == e.key)
                    {
                        roles.push((d.name.clone(), r.id.clone(), r.state.clone()));
                    }
                }
                let p = &e.profile;
                MachineView {
                    key: e.key.clone(),
                    name: p.name.clone(),
                    chip: p.chip.clone(),
                    backend: p.backend.label().into(),
                    os: p.os.clone(),
                    cpu_cores: p.cpu_cores,
                    gpu_cores: p.gpu_cores,
                    total_memory: p.accel_memory.unwrap_or(p.total_memory).0,
                    usable_memory: p.usable_memory.0,
                    free_memory: p.usable_memory.0.saturating_sub(used),
                    bandwidth_gbs: p.effective_bandwidth_gbs(),
                    tflops: p.effective_tflops(),
                    measured: p.measured_bandwidth_gbs.is_some(),
                    local: e.local,
                    address: e.address.clone(),
                    rtt_ms: e.rtt_ms,
                    on_battery: p.on_battery,
                    engines: e.engines.clone(),
                    roles,
                }
            })
            .collect()
    }

    pub fn machine_count(&self) -> usize {
        self.machines.lock().unwrap().len()
    }

    /// Wait until machines, engines or deployments change.
    pub async fn changed(&self) {
        self.changed.notified().await
    }

    /// Ask every machine (or one) to look for engines again.
    pub fn redetect(&self, key: Option<&str>) {
        for e in self.machines.lock().unwrap().values() {
            if key.is_none_or(|k| k == e.key || k == e.profile.name) {
                let _ = e.tx.send(Msg::Detect);
            }
        }
    }

    /// Send a machine away (it exits).
    pub fn remove_machine(&self, key: &str) -> Result<String> {
        let e = {
            let mut ms = self.machines.lock().unwrap();
            let k = ms
                .values()
                .find(|e| e.key == key || e.profile.name == key)
                .map(|e| e.key.clone())
                .with_context(|| format!("no machine {key}"))?;
            if ms[&k].local {
                bail!("this server's own machine can't be removed — stop the server instead");
            }
            ms.remove(&k).unwrap()
        };
        let _ = e.tx.send(Msg::Bye {
            reason: "removed from the cluster".into(),
            leave: true,
        });
        self.machine_lost(&e.key, &e.profile.name);
        self.bump();
        self.event("info", format!("Removed {}", e.profile.name), None);
        Ok(e.profile.name)
    }

    // ---- planning -------------------------------------------------------

    async fn resolve(&self, engine: &Arc<dyn Engine>, req: &ModelRequest) -> Result<ResolvedModel> {
        let key = format!(
            "{}|{}|{:?}|{}|{}",
            engine.info().id,
            req.model,
            req.quantize,
            req.context,
            req.concurrency
        );
        if let Some((t, m)) = self.resolved.lock().unwrap().get(&key) {
            if t.elapsed() < Duration::from_secs(600) {
                return Ok(m.clone());
            }
        }
        let m = engine.resolve(req).await?;
        self.resolved
            .lock()
            .unwrap()
            .insert(key, (Instant::now(), m.clone()));
        Ok(m)
    }

    /// What loading `req` would do (nothing is started).
    pub async fn plan(&self, req: &LoadRequest) -> Result<PlanPreview> {
        Ok(self.plan_full(req).await?.0)
    }

    async fn plan_full(&self, req: &LoadRequest) -> Result<Planned> {
        let engine = tendril_engines::get(&req.engine).with_context(|| {
            format!(
                "unknown engine '{}' — one of: {}",
                req.engine,
                tendril_engines::all()
                    .iter()
                    .map(|e| e.info().id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let info = engine.info();
        let version = self.version();
        let default_conc = match info.tasks.first() {
            Some(Task::Chat) | None => self.opts.default_concurrency,
            _ => 1,
        };
        let mreq = ModelRequest {
            model: req.model.trim().to_string(),
            quantize: req.quantize.clone().filter(|q| !q.is_empty()),
            context: req
                .context
                .unwrap_or(self.opts.default_context)
                .clamp(256, 1 << 21),
            concurrency: req.concurrency.unwrap_or(default_conc).clamp(1, 256),
        };
        let model = self.resolve(&engine, &mreq).await?;
        let context = if model.spec.max_position > 0 && req.context.is_none() {
            mreq.context.min(model.spec.max_position as usize)
        } else {
            mreq.context
        };
        let concurrency = mreq.concurrency;

        // Machines, what they have free, and whether they can run the engine.
        let deps = self.deployments();
        let mut machines = Vec::new();
        let mut profiles: Vec<NodeProfile> = Vec::new();
        let mut unavailable = Vec::new();
        for e in self.machines.lock().unwrap().values() {
            if let Some(only) = &req.machines {
                if !only.iter().any(|o| *o == e.key || *o == e.profile.name) {
                    continue;
                }
            }
            let det = e.detection(&info.id);
            if !det.ready {
                unavailable.push(Unavailable {
                    machine: e.profile.name.clone(),
                    missing: det.missing.clone(),
                    notes: det.notes.clone(),
                });
                continue;
            }
            let used: u64 = deps
                .iter()
                .flat_map(|d| d.budgets.iter())
                .filter(|(k, _)| *k == e.key)
                .map(|(_, b)| *b)
                .sum();
            let mut p = e.profile.clone();
            p.usable_memory = Bytes(p.usable_memory.0.saturating_sub(used));
            machines.push(Machine {
                key: e.key.clone(),
                name: e.profile.name.clone(),
                address: e.address.clone(),
                detection: det,
                local: e.local,
            });
            profiles.push(p);
        }
        let mut preview = PlanPreview {
            pool_version: version,
            engine: info.clone(),
            model: model.id.clone(),
            task: model.task,
            vision: model.vision,
            weight_bytes: model.spec.weight_bytes().0,
            layers: model.spec.num_layers,
            layered: model.layered,
            context,
            concurrency,
            fits: false,
            reason: None,
            stages: vec![],
            tokens_per_sec: None,
            ttft_ms: None,
            needs_trusted_network: false,
            security: String::new(),
            risk: None,
            unavailable,
            notes: model.notes.clone(),
            commands: vec![],
        };
        let fail = |mut p: PlanPreview, why: String| -> Result<Planned> {
            p.reason = Some(why);
            Ok((p, model.clone(), vec![], vec![], engine.clone()))
        };
        if machines.is_empty() {
            let why = if self.machine_count() == 0 {
                "No machines in the cluster yet.".to_string()
            } else {
                format!(
                    "No machine has {} ready — install it (see below), then press “Check again”.",
                    info.name
                )
            };
            return fail(preview, why);
        }
        let multi = match info.multi_machine {
            MultiMachine::Single => false,
            MultiMachine::EngineManaged => true,
            _ => model.layered,
        };
        // Which machines may take part: heads always, workers when splitting.
        let mut idx: Vec<usize> = (0..machines.len())
            .filter(|&i| {
                engine.can_head(&machines[i].detection)
                    || (multi && engine.can_join(&machines[i].detection))
            })
            .collect();
        if model.local_path.is_some() && !multi {
            idx.retain(|&i| machines[i].local);
            if idx.is_empty() {
                return fail(preview, "The model is a path on this server, so it must run here — but this server doesn't share its compute or lacks the engine.".into());
            }
        }
        if idx.is_empty() {
            return fail(
                preview,
                format!(
                    "No machine can serve with {} here (see the notes per machine).",
                    info.name
                ),
            );
        }
        let mut peaks: HashMap<String, u64> = HashMap::new();
        let (stages, tps, ttft) = if info.multi_machine == MultiMachine::EngineManaged {
            // The engine splits by itself: every machine takes part, sized by
            // its free memory.
            let free: u64 = idx.iter().map(|&i| profiles[i].usable_memory.0).sum();
            let need = model.spec.weight_bytes().0;
            if need > 0 && free < need + need / 10 {
                return fail(
                    preview,
                    format!(
                        "{} needs about {} but the machines with {} have {} free",
                        model.id,
                        Bytes(need + need / 10),
                        info.name,
                        Bytes(free)
                    ),
                );
            }
            let mut stages = Vec::new();
            for &i in &idx {
                let share = if free > 0 {
                    need as f64 * profiles[i].usable_memory.0 as f64 / free as f64
                } else {
                    0.0
                };
                stages.push(Stage {
                    machine: machines[i].key.clone(),
                    layers: (0, 0),
                    weight_bytes: share as u64,
                });
                peaks.insert(machines[i].key.clone(), (share * 1.1) as u64);
            }
            (stages, None, None)
        } else {
            let link = Link::preset("1gbe").expect("preset");
            let workload = Workload::new(context as u64, concurrency as u64);
            let run = |sub: Vec<usize>, max: usize, min: usize| {
                let cluster = Cluster::new(
                    sub.iter().map(|&i| profiles[i].clone()).collect(),
                    link.clone(),
                );
                let opts = PlanOptions {
                    max_stages: max,
                    min_stages: min,
                    ..Default::default()
                };
                let (spec, w) = (model.spec.clone(), workload.clone());
                async move {
                    let r =
                        tokio::task::spawn_blocking(move || plan(&spec, &cluster, &w, &opts)).await;
                    (sub, r)
                }
            };
            let min = if multi {
                req.min_machines.unwrap_or(1).max(1)
            } else {
                1
            };
            // A model file on this server is served from here: try this
            // machine alone first, else a split that includes it.
            let local: Vec<usize> = idx.iter().copied().filter(|&i| machines[i].local).collect();
            let mut attempt = None;
            if model.local_path.is_some() && !local.is_empty() && min <= 1 {
                let (sub, r) = run(local.clone(), 1, 1).await;
                if let Ok(r) = r {
                    if r.selected.is_some() {
                        attempt = Some((sub, r));
                    }
                }
            }
            let (sub, r) = match attempt {
                Some(a) => a,
                None => {
                    let (sub, r) = run(idx.clone(), if multi { 8 } else { 1 }, min).await;
                    (sub, r?)
                }
            };
            let Some(p) = r.selected else {
                let why = r
                    .rejected
                    .first()
                    .map(|x| x.reason.clone())
                    .unwrap_or_else(|| {
                        format!(
                            "no placement on {} machine(s) that can run {}{}",
                            sub.len(),
                            info.name,
                            if min > 1 {
                                format!(" (asked for at least {min})")
                            } else {
                                String::new()
                            }
                        )
                    });
                return fail(preview, why);
            };
            let idx = sub;
            let mut stages = Vec::new();
            for st in &p.stages {
                let m = &machines[idx[st.node]];
                stages.push(Stage {
                    machine: m.key.clone(),
                    layers: (st.layer_start, st.layer_end),
                    weight_bytes: st.mem.weights.0,
                });
                peaks.insert(m.key.clone(), st.mem.peak.0);
            }
            let layered = model.layered;
            (
                stages,
                layered.then_some(p.tokens_per_sec),
                layered.then_some(p.ttft_ms),
            )
        };
        let placement = Placement {
            stages,
            head: 0,
            total_layers: model.spec.num_layers,
            context,
            concurrency,
        };
        let placement = match engine.arrange(&model, placement, &machines) {
            Ok(p) => p,
            Err(e) => return fail(preview, format!("{e:#}")),
        };
        let roles = match engine.roles(&model, &placement, &machines) {
            Ok(r) => r,
            Err(e) => return fail(preview, format!("{e:#}")),
        };
        let at = |k: &str| machines.iter().position(|m| m.key == k);
        let total_w: u64 = placement
            .stages
            .iter()
            .map(|s| s.weight_bytes)
            .sum::<u64>()
            .max(1);
        for (i, s) in placement.stages.iter().enumerate() {
            let m = at(&s.machine);
            preview.stages.push(StageView {
                machine: m
                    .map(|j| machines[j].name.clone())
                    .unwrap_or_else(|| s.machine.clone()),
                key: s.machine.clone(),
                backend: m
                    .map(|j| profiles[j].backend.label().to_string())
                    .unwrap_or_default(),
                layers: s.layers,
                weight_bytes: s.weight_bytes,
                weight_pct: 100.0 * s.weight_bytes as f64 / total_w as f64,
                memory_needed: peaks.get(&s.machine).copied().unwrap_or(s.weight_bytes),
                memory_free: m.map(|j| profiles[j].usable_memory.0).unwrap_or(0),
                serves_api: i == placement.head,
                transfer: engine.transfer_note(&model, &placement, i),
            });
        }
        let spans = placement.stages.len() > 1;
        preview.needs_trusted_network = spans && info.security.needs_trusted_network();
        (preview.security, preview.risk) = match (&info.security, spans) {
            (_, false) | (Security::LocalOnly, _) => (
                "Runs on one machine; Tendril carries its API to the server over an encrypted, token-authenticated link.".into(),
                None,
            ),
            (Security::TendrilTunnel { what }, true) => (what.clone(), None),
            (Security::EngineEncrypted { what }, true) => (what.clone(), None),
            (Security::TrustedNetworkOnly { what, risk }, true) => (what.clone(), Some(risk.clone())),
        };
        preview.tokens_per_sec = tps;
        preview.ttft_ms = ttft;
        preview.commands = roles
            .iter()
            .map(|r| {
                let n = at(&r.machine)
                    .map(|j| machines[j].name.clone())
                    .unwrap_or_default();
                (n, describe_role(r))
            })
            .collect();
        preview.fits = true;
        let budgets: Vec<(String, u64)> = peaks.into_iter().map(|(k, p)| (k, budget(p))).collect();
        Ok((preview, model, roles, budgets, engine))
    }

    // ---- deployments ----------------------------------------------------

    /// Load a model; returns its name. The model keeps loading in the background.
    pub async fn load(self: &Arc<Self>, req: &LoadRequest) -> Result<String> {
        if self.stopping.load(Ordering::Relaxed) {
            bail!("the cluster is stopping");
        }
        let (preview, model, roles, budgets, engine) = self.plan_full(req).await?;
        if req
            .expect_version
            .is_some_and(|v| v != preview.pool_version)
        {
            return Err(PoolChanged.into());
        }
        if !preview.fits {
            bail!(
                "{}",
                preview
                    .reason
                    .clone()
                    .unwrap_or_else(|| "it doesn't fit".into())
            );
        }
        if preview.needs_trusted_network && !req.trusted_network {
            return Err(NeedsTrust(format!(
                "{} sends this model's traffic between machines unencrypted. Confirm the machines share a private network you trust (trusted_network: true), or load it on one machine.",
                preview.engine.name
            ))
            .into());
        }
        let name = model.served_name.clone();
        let id = format!("d{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let names: HashMap<String, String> = self
            .machines
            .lock()
            .unwrap()
            .values()
            .map(|e| (e.key.clone(), e.profile.name.clone()))
            .collect();
        let views = roles
            .iter()
            .map(|r| RoleView {
                id: r.id.clone(),
                machine: r.machine.clone(),
                machine_name: names.get(&r.machine).cloned().unwrap_or_default(),
                serves_api: r.serves_api,
                state: "waiting".into(),
                detail: String::new(),
                done: 0,
                total: 0,
                sent: false,
                ports: None,
                ready: false,
                log: vec![],
            })
            .collect();
        let dep = Arc::new(Deployment {
            id: id.clone(),
            name: name.clone(),
            engine,
            model,
            roles,
            budgets,
            preview: preview.clone(),
            started: Instant::now(),
            state: Mutex::new(DepState {
                status: Status::Loading,
                reason: None,
                roles: views,
                api: None,
            }),
            changed: Notify::new(),
            requests: AtomicU64::new(0),
        });
        {
            let mut deps = self.deployments.lock().unwrap();
            if deps.iter().any(|d| d.name == name) {
                bail!("{name} is already loaded — unload it first");
            }
            deps.push(dep.clone());
        }
        self.bump();
        self.event(
            "info",
            format!(
                "Loading {name} with {} on {}",
                preview.engine.name,
                preview
                    .stages
                    .iter()
                    .map(|s| s.machine.clone())
                    .collect::<Vec<_>>()
                    .join(" + ")
            ),
            Some(&name),
        );
        tokio::spawn(self.clone().drive(dep));
        Ok(name)
    }

    /// Start the deployment's roles in order, then expose its API.
    async fn drive(self: Arc<Self>, dep: Arc<Deployment>) {
        loop {
            let notified = dep.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if matches!(dep.status(), Status::Failed | Status::Stopping) {
                return;
            }
            let mut to_start = Vec::new();
            {
                let mut st = dep.state.lock().unwrap();
                let snapshot = st.roles.clone();
                let ready = |id: &str| snapshot.iter().any(|v| v.id == id && v.ready);
                let started = |id: &str| snapshot.iter().any(|v| v.id == id && v.ports.is_some());
                for (i, r) in dep.roles.iter().enumerate() {
                    if !st.roles[i].sent
                        && r.after.iter().all(|a| ready(a))
                        && r.after_started.iter().all(|a| started(a))
                    {
                        st.roles[i].sent = true;
                        to_start.push(i);
                    }
                }
            }
            for i in to_start {
                if let Err(e) = self.send_start(&dep, i) {
                    self.fail(&dep, format!("{e:#}"));
                    return;
                }
            }
            let api_ready = {
                let st = dep.state.lock().unwrap();
                st.status == Status::Loading && st.roles.iter().any(|r| r.serves_api && r.ready)
            };
            if api_ready {
                match self.api_route(&dep).await {
                    Ok(f) => {
                        {
                            let mut st = dep.state.lock().unwrap();
                            if st.status != Status::Loading {
                                return;
                            }
                            st.api = Some(f);
                            st.status = Status::Ready;
                        }
                        self.bump();
                        self.event(
                            "ok",
                            format!(
                                "{} is ready ({}, {:.0} s)",
                                dep.name,
                                dep.engine.info().name,
                                dep.started.elapsed().as_secs_f64()
                            ),
                            Some(&dep.name),
                        );
                    }
                    Err(e) => {
                        self.fail(&dep, format!("cannot reach the engine's API: {e:#}"));
                        return;
                    }
                }
            }
            let _ = tokio::time::timeout(Duration::from_secs(5), notified).await;
        }
    }

    fn send_start(&self, dep: &Deployment, i: usize) -> Result<()> {
        let role = &dep.roles[i];
        let ms = self.machines.lock().unwrap();
        let m = ms
            .get(&role.machine)
            .context("its machine left the cluster")?;
        let peers: Vec<Peer> = {
            let st = dep.state.lock().unwrap();
            st.roles
                .iter()
                .filter_map(|v| {
                    let ports = v.ports.clone()?;
                    let pm = ms.get(&v.machine)?;
                    Some(Peer {
                        role: v.id.clone(),
                        tunnel: pm.tunnel(),
                        host: pm.address.clone(),
                        ports,
                    })
                })
                .collect()
        };
        let namespace = format!(
            "tendril-{}-{}",
            crate::discovery::fingerprint(&self.opts.token),
            dep.id
        );
        m.tx.send(Msg::Start {
            deployment: dep.id.clone(),
            role: serde_json::to_string(role)?,
            peers,
            namespace,
        })
        .map_err(|_| anyhow::anyhow!("lost the connection to {}", m.profile.name))?;
        Ok(())
    }

    async fn api_route(&self, dep: &Deployment) -> Result<Forwarder> {
        let route = {
            let st = dep.state.lock().unwrap();
            let r = st
                .roles
                .iter()
                .find(|r| r.serves_api)
                .context("no API role")?;
            let port = r
                .ports
                .as_ref()
                .and_then(|p| p.iter().find(|(k, _)| k.is_empty()).map(|(_, v)| *v))
                .context("the API role reported no port")?;
            let ms = self.machines.lock().unwrap();
            let m = ms.get(&r.machine).context("its machine left")?;
            if m.local {
                Route::Local(port)
            } else {
                Route::Remote {
                    tunnel: m.tunnel(),
                    target: format!("{}/{}", dep.id, r.id),
                }
            }
        };
        Forwarder::start(route, self.psk).await
    }

    fn role_update(&self, deployment: &str, role: &str, state: RoleState) {
        let Some(dep) = self.deployments().into_iter().find(|d| d.id == deployment) else {
            return;
        };
        let mut failed = None;
        {
            let mut st = dep.state.lock().unwrap();
            let Some(v) = st.roles.iter_mut().find(|r| r.id == role) else {
                return;
            };
            match state {
                RoleState::Downloading { file, done, total } => {
                    v.state = "downloading".into();
                    v.detail = file;
                    (v.done, v.total) = (done, total);
                }
                RoleState::Starting => {
                    v.state = "starting".into();
                    v.detail = String::new();
                    (v.done, v.total) = (0, 0);
                }
                RoleState::Started { ports } => {
                    v.state = "loading".into();
                    v.ports = Some(ports);
                }
                RoleState::Preparing { what, done, total } => {
                    v.state = "preparing".into();
                    v.detail = what;
                    (v.done, v.total) = (done, total);
                }
                RoleState::Ready => {
                    v.state = "ready".into();
                    v.detail = String::new();
                    v.ready = true;
                }
                RoleState::Failed { error, log } => {
                    v.state = "failed".into();
                    v.detail = error.clone();
                    v.log = log;
                    failed = Some(format!("{} on {}: {error}", v.id, v.machine_name));
                }
                RoleState::Exited { code, log } => {
                    v.state = "exited".into();
                    v.detail = format!(
                        "exit code {}",
                        code.map(|c| c.to_string()).unwrap_or_else(|| "?".into())
                    );
                    v.log = log;
                    failed = Some(format!(
                        "{} on {} stopped ({})",
                        v.id, v.machine_name, v.detail
                    ));
                }
            }
        }
        if let Some(why) = failed {
            self.fail(&dep, why);
        }
        dep.changed.notify_waiters();
        self.changed.notify_waiters();
    }

    /// Mark a deployment failed and stop its processes everywhere.
    fn fail(&self, dep: &Deployment, why: String) {
        {
            let mut st = dep.state.lock().unwrap();
            if matches!(st.status, Status::Failed | Status::Stopping) {
                return;
            }
            st.status = Status::Failed;
            st.reason = Some(why.clone());
            st.api = None;
        }
        self.stop_roles(dep);
        dep.changed.notify_waiters();
        self.bump();
        self.event(
            "error",
            format!("{} failed: {why}", dep.name),
            Some(&dep.name),
        );
    }

    fn stop_roles(&self, dep: &Deployment) {
        let ms = self.machines.lock().unwrap();
        for r in &dep.roles {
            if let Some(m) = ms.get(&r.machine) {
                let _ = m.tx.send(Msg::Stop {
                    deployment: dep.id.clone(),
                    role: r.id.clone(),
                });
            }
        }
    }

    /// Stop a model and free its memory.
    pub fn unload(&self, name: &str) -> Result<String> {
        let dep = {
            let mut deps = self.deployments.lock().unwrap();
            let i = deps
                .iter()
                .position(|d| d.name == name || d.id == name)
                .with_context(|| format!("no model named {name}"))?;
            deps.remove(i)
        };
        {
            let mut st = dep.state.lock().unwrap();
            st.status = Status::Stopping;
            st.api = None;
        }
        self.stop_roles(&dep);
        dep.changed.notify_waiters();
        self.bump();
        self.event("info", format!("Unloaded {}", dep.name), Some(&dep.name));
        Ok(dep.name.clone())
    }

    pub fn deployments(&self) -> Vec<Arc<Deployment>> {
        self.deployments.lock().unwrap().clone()
    }

    /// The deployment a request is for: by name, or the only one for `task`.
    pub fn target(&self, model: Option<&str>, task: Task) -> Result<Arc<Deployment>, String> {
        let deps = self.deployments();
        if let Some(m) = model.filter(|m| !m.is_empty()) {
            if let Some(d) = deps
                .iter()
                .find(|d| d.name == m || d.model.id == m || d.id == m)
            {
                return Ok(d.clone());
            }
            // Tolerate clients that send a placeholder name when only one model is loaded.
            if let [d] = deps.as_slice() {
                if d.model.task == task {
                    return Ok(d.clone());
                }
            }
            return Err(format!(
                "no model named '{m}' — loaded: {}",
                list_or_none(&deps.iter().map(|d| d.name.clone()).collect::<Vec<_>>())
            ));
        }
        let fit: Vec<_> = deps.iter().filter(|d| d.model.task == task).collect();
        match fit.as_slice() {
            [d] => Ok((*d).clone()),
            [] => Err("no model for this kind of request is loaded — load one in the dashboard or with `tendril load`".into()),
            _ => Err(format!(
                "several models are loaded; say which with \"model\": {}",
                fit.iter().map(|d| d.name.clone()).collect::<Vec<_>>().join(", ")
            )),
        }
    }

    // ---- stopping -------------------------------------------------------

    /// The dashboard's "Stop cluster".
    pub fn request_stop(&self) {
        self.stop.notify_one();
    }

    pub async fn stopped(&self) {
        self.stop.notified().await
    }

    /// Stop every model and send every machine away.
    pub async fn shutdown(&self) {
        self.stopping.store(true, Ordering::Relaxed);
        let names: Vec<String> = self.deployments().iter().map(|d| d.name.clone()).collect();
        for n in names {
            let _ = self.unload(&n);
        }
        for e in self.machines.lock().unwrap().values() {
            let _ = e.tx.send(Msg::Bye {
                reason: "the server stopped the cluster".into(),
                leave: true,
            });
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    /// "ready" / "loading" / "empty", for discovery beacons.
    pub fn state_word(&self) -> &'static str {
        let deps = self.deployments();
        if deps.iter().any(|d| d.status() == Status::Ready) {
            "ready"
        } else if deps.iter().any(|d| d.status() == Status::Loading) {
            "loading"
        } else {
            "empty"
        }
    }
}

fn describe_role(r: &Role) -> String {
    let l = &r.launch;
    let mut parts = vec![l.program.clone()];
    parts.extend(l.args.iter().cloned());
    let cmd = tendril_engines::util::command_line(&parts);
    let env: Vec<String> = l.env.iter().map(|(k, v)| format!("{k}={v}")).collect();
    if env.is_empty() {
        cmd
    } else {
        format!("{} {cmd}", env.join(" "))
    }
}

fn list_or_none(v: &[String]) -> String {
    if v.is_empty() {
        "none".into()
    } else {
        v.join(", ")
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}
