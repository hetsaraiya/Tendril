//! Several models served from one pool of machines.
//!
//! Each model runs its own coordinator (pipeline, batching, caches). The pool
//! owns the control port: every joining machine opens one session per model,
//! all tagged with the same machine id, and the pool splits each machine's
//! memory between the models with `tendril_core::pool::allocate`. A model's
//! coordinator then plans inside its share like it would on a smaller machine.

use crate::coordinator::{
    model_spec, read_hello, Coordinator, Event, PoolLink, ServeOptions, Status, LOCAL_MACHINE,
};
use crate::proto::Msg;
use crate::{token, wire};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tendril_core::cluster::{Cluster, Link};
use tendril_core::hardware::NodeProfile;
use tendril_core::model::ModelSpec;
use tendril_core::planner::{Plan, PlanOptions, Workload};
use tendril_core::pool::{allocate, PoolModel};
use tendril_core::units::Bytes;
use tendril_engine::linear::WeightFormat;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, mpsc, Notify};

/// What to load: a model plus optional overrides of the server's defaults.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LoadRequest {
    pub model: String,
    /// native, q8_0, q6_k or q4_k.
    #[serde(default)]
    pub quantize: Option<String>,
    /// Tokens per conversation.
    #[serde(default)]
    pub context: Option<usize>,
}

/// Turns a load request into serve options (fetching the model's config
/// and tensor table, not its weights).
pub type Resolver = Arc<dyn Fn(&LoadRequest) -> Result<ServeOptions> + Send + Sync>;

/// A machine waiting in the pool (its first session).
struct LobbyEntry {
    profile: NodeProfile,
    tx: mpsc::UnboundedSender<Msg>,
    peer: IpAddr,
    rtt_ms: Option<f64>,
    gbps: Option<f64>,
}

/// One machine of the pool, for the dashboard.
#[derive(Clone, Debug, Serialize)]
pub struct MachineView {
    /// Stable id for control requests ("local" for this server).
    pub id: String,
    pub name: String,
    pub chip: String,
    pub backend: String,
    pub os: String,
    pub cpu_cores: u32,
    pub gpu_cores: Option<u32>,
    pub total_memory: u64,
    pub usable_memory: u64,
    pub available_memory: Option<u64>,
    /// Measured by the startup benchmark (None: spec-sheet estimate below).
    pub weight_gbs: Option<f64>,
    pub tflops: Option<f64>,
    pub spec_bandwidth_gbs: f64,
    pub spec_tflops: f64,
    pub local: bool,
    pub address: String,
    pub rtt_ms: Option<f64>,
    pub bandwidth_gbps: Option<f64>,
    pub on_battery: Option<bool>,
    /// What each model runs here.
    pub roles: Vec<MachineRole>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MachineRole {
    pub model: String,
    /// "layers 0–15 + embeddings", "idle", …
    pub role: String,
    /// Memory this model may use here.
    pub budget: u64,
}

/// What loading a model would do, before anything is downloaded.
#[derive(Clone, Debug, Serialize)]
pub struct PlanPreview {
    pub model: String,
    pub architecture: String,
    pub params: u64,
    pub layers: u64,
    pub format: String,
    pub weight_bytes: u64,
    pub context: usize,
    pub concurrency: usize,
    pub machines: usize,
    pub fits: bool,
    pub reason: Option<String>,
    /// What each machine gets, in pipeline order.
    pub shares: Vec<DeviceShare>,
    /// Predicted speed of the split (None if it doesn't fit).
    pub tokens_per_sec: Option<f64>,
    pub ttft_ms: Option<f64>,
    /// The full plan (node indices refer to `machine_names`).
    pub plan: Option<Plan>,
    pub machine_names: Vec<String>,
    /// Loaded models that would no longer fit next to this one.
    pub displaces: Vec<String>,
    /// Same model in other weight formats.
    pub alternatives: Vec<Alternative>,
    pub advice: Vec<String>,
}

/// One machine's part of a planned model.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceShare {
    pub machine: String,
    pub backend: String,
    /// Transformer layers [start, end).
    pub layers: (u64, u64),
    /// "embeddings, layers 0–15", …
    pub components: String,
    /// Weights this machine holds — and downloads, only these byte ranges.
    pub weight_bytes: u64,
    /// Share of the model's weights, 0–100.
    pub weight_pct: f64,
    /// Peak memory it needs (weights + KV + buffers) and what it may use.
    pub memory_needed: u64,
    pub memory_budget: u64,
    pub kv_bytes: u64,
    /// Its part of each decode step, ms.
    pub decode_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Alternative {
    pub format: String,
    pub weight_bytes: u64,
    pub fits: bool,
    pub tokens_per_sec: Option<f64>,
}

/// One model's slice of the pool, for status pages.
#[derive(Clone, Debug, Serialize)]
pub struct Share {
    pub model: String,
    pub placed: bool,
    pub reason: Option<String>,
    /// (machine name, memory this model may use there).
    pub machines: Vec<(String, u64)>,
}

pub struct Pool {
    models: StdMutex<Vec<Coordinator>>,
    links: StdMutex<Vec<Arc<PoolLink>>>,
    shares: StdMutex<Vec<Share>>,
    events: broadcast::Sender<Event>,
    history: Arc<StdMutex<VecDeque<Event>>>,
    changed: Arc<Notify>,
    /// Each joined machine's first session, by machine id: machines wait
    /// here for models and hear about new ones.
    lobby: StdMutex<HashMap<u64, LobbyEntry>>,
    adding: tokio::sync::Mutex<()>,
    /// This server's own machine, when it shares its compute.
    local: Option<NodeProfile>,
    /// Recently planned requests, so confirming doesn't fetch them again.
    resolved: StdMutex<HashMap<String, (Instant, ServeOptions)>>,
    /// Set by the dashboard's "Stop cluster".
    stop: Notify,
    control_port: u16,
    /// The cluster token (also guards the model-loading API).
    pub token: String,
    resolve: Option<Resolver>,
}

impl Pool {
    /// Start serving `opts` (one entry per model; the first one's ports,
    /// token and machine settings apply to the pool).
    pub async fn start(opts: Vec<ServeOptions>) -> Result<Arc<Pool>> {
        let first = opts.first().context("no model to serve")?;
        let local = if first.use_local {
            Some(crate::coordinator::this_machine(
                first.name.as_deref(),
                &first.device,
                first.max_memory,
            )?)
        } else {
            None
        };
        let pool = Pool::open(first.control_port, first.token.clone(), None, local).await?;
        for o in opts {
            pool.add(o).await?;
        }
        Ok(pool)
    }

    /// An empty pool: machines can join now, models are added later.
    pub async fn open(
        control_port: u16,
        token: String,
        resolve: Option<Resolver>,
        local: Option<NodeProfile>,
    ) -> Result<Arc<Pool>> {
        let control = TcpListener::bind(("0.0.0.0", control_port))
            .await
            .with_context(|| {
                format!("cannot listen on port {control_port} (is another Tendril running?)")
            })?;
        let (events, _) = broadcast::channel(512);
        let changed = Arc::new(Notify::new());
        let pool = Arc::new(Pool {
            models: StdMutex::new(vec![]),
            links: StdMutex::new(vec![]),
            shares: StdMutex::new(vec![]),
            events,
            history: Arc::new(StdMutex::new(VecDeque::new())),
            changed: changed.clone(),
            lobby: StdMutex::new(HashMap::new()),
            adding: tokio::sync::Mutex::new(()),
            control_port,
            token: token.clone(),
            resolve,
            local,
            resolved: StdMutex::new(HashMap::new()),
            stop: Notify::new(),
        });
        tokio::spawn(pool.clone().control_loop(control, token::psk(&token)));
        tokio::spawn(pool.clone().allocator_loop(changed));
        Ok(pool)
    }

    /// Resolve a request (config and tensor table, no weights), reusing a
    /// resolution from a recent plan preview.
    async fn resolve(&self, req: &LoadRequest) -> Result<ServeOptions> {
        let key = serde_json::to_string(req)?;
        {
            let mut cache = self.resolved.lock().unwrap();
            cache.retain(|_, (at, _)| at.elapsed() < Duration::from_secs(600));
            if let Some((_, o)) = cache.get(&key) {
                return Ok(o.clone());
            }
        }
        let resolve = self
            .resolve
            .clone()
            .context("this server can't load models on request")?;
        let r = req.clone();
        let opts = tokio::task::spawn_blocking(move || resolve(&r)).await??;
        self.resolved
            .lock()
            .unwrap()
            .insert(key, (Instant::now(), opts.clone()));
        Ok(opts)
    }

    /// Resolve `name` with the server's defaults and serve it.
    pub async fn load(&self, name: &str) -> Result<String> {
        self.load_with(&LoadRequest {
            model: name.to_string(),
            ..Default::default()
        })
        .await
    }

    /// Resolve a request and serve it on the pool's machines.
    pub async fn load_with(&self, req: &LoadRequest) -> Result<String> {
        let opts = self.resolve(req).await?;
        let model = opts.model_name.clone();
        self.add(opts).await?;
        Ok(model)
    }

    /// Serve one more model on the pool's machines.
    pub async fn add(&self, mut opts: ServeOptions) -> Result<()> {
        let _one_at_a_time = self.adding.lock().await;
        let mut names = self.names();
        if names.contains(&opts.model_name) {
            bail!("{} is already loaded", opts.model_name);
        }
        names.push(opts.model_name.clone());
        opts.control_port = self.control_port;
        let link = Arc::new(PoolLink {
            index: AtomicUsize::new(names.len() - 1),
            models: StdMutex::new(names.clone()),
            budgets: StdMutex::new(None),
            plan: StdMutex::new(None),
            note: StdMutex::new(None),
            changed: self.changed.clone(),
        });
        let c = Coordinator::start_with(opts, Some(link.clone())).await?;
        self.forward_events(&c);
        self.shares.lock().unwrap().push(Share {
            model: c.inner.opts.model_name.clone(),
            placed: false,
            reason: None,
            machines: vec![],
        });
        self.models.lock().unwrap().push(c);
        self.links.lock().unwrap().push(link);
        self.announce_models();
        self.changed.notify_one();
        Ok(())
    }

    /// Tell every model's coordinator and every joined machine which models
    /// the pool serves (machines open a session for each new one).
    fn announce_models(&self) {
        let names = self.names();
        for (i, l) in self.links.lock().unwrap().iter().enumerate() {
            l.index.store(i, Ordering::Relaxed);
            *l.models.lock().unwrap() = names.clone();
        }
        for e in self.lobby.lock().unwrap().values() {
            let _ = e.tx.send(Msg::Models {
                models: names.clone(),
            });
        }
    }

    /// Stop serving a model; machines keep its memory free for others.
    pub async fn unload(&self, name: &str) -> Result<String> {
        let _one_at_a_time = self.adding.lock().await;
        let c = self.get(Some(name)).map_err(|e| anyhow::anyhow!(e))?;
        let model = c.inner.opts.model_name.clone();
        {
            let mut models = self.models.lock().unwrap();
            let i = models
                .iter()
                .position(|m| m.inner.opts.model_name == model)
                .context("not loaded")?;
            models.remove(i);
            self.links.lock().unwrap().remove(i);
            self.shares.lock().unwrap().retain(|s| s.model != model);
        }
        c.shutdown(false).await;
        self.announce_models();
        self.event("info", format!("Unloaded {model}"));
        self.changed.notify_one();
        Ok(model)
    }

    /// Remove a joined machine: it unloads everything and exits.
    pub fn remove_machine(&self, id: &str) -> Result<String> {
        let machine = u64::from_str_radix(id, 16).context("unknown machine")?;
        if machine == LOCAL_MACHINE {
            bail!("this server can't remove itself — stop it, or start it with --no-local");
        }
        let reason = "the host removed this machine from the cluster";
        let name = {
            let lobby = self.lobby.lock().unwrap();
            let e = lobby.get(&machine).context("no such machine in the pool")?;
            let _ = e.tx.send(Msg::Bye {
                reason: reason.into(),
                leave: true,
            });
            e.profile.name.clone()
        };
        for c in self.models() {
            c.kick(machine, reason);
        }
        self.event("warn", format!("Removed {name} from the cluster"));
        Ok(name)
    }

    /// Ask the server to stop (the dashboard's "Stop cluster").
    pub fn request_stop(&self) {
        self.stop.notify_one();
    }

    /// Resolves when someone asked the server to stop.
    pub async fn stopped(&self) {
        self.stop.notified().await
    }

    pub fn models(&self) -> Vec<Coordinator> {
        self.models.lock().unwrap().clone()
    }

    pub fn names(&self) -> Vec<String> {
        self.models()
            .iter()
            .map(|c| c.inner.opts.model_name.clone())
            .collect()
    }

    /// Machines waiting in (or joined to) the pool, as the lobby sees them.
    pub fn machines(&self) -> Vec<NodeProfile> {
        self.lobby
            .lock()
            .unwrap()
            .values()
            .map(|e| e.profile.clone())
            .collect()
    }

    /// Every machine of the pool (this server first) with its compute,
    /// links and what each model runs on it.
    pub fn machine_views(&self) -> Vec<MachineView> {
        let models = self.models();
        let shares = self.shares();
        // Per model: machine id → role, from each coordinator's view.
        let roles: Vec<(String, HashMap<String, String>)> = models
            .iter()
            .map(|c| {
                let map = c
                    .nodes()
                    .into_iter()
                    .map(|n| (n.machine.clone(), n.role))
                    .collect();
                (c.inner.opts.model_name.clone(), map)
            })
            .collect();
        let view = |id: String,
                    p: &NodeProfile,
                    local: bool,
                    address: String,
                    rtt_ms: Option<f64>,
                    gbps: Option<f64>| {
            let roles = roles
                .iter()
                .map(|(model, map)| MachineRole {
                    model: model.clone(),
                    role: map.get(&id).cloned().unwrap_or_else(|| "not used".into()),
                    budget: shares
                        .iter()
                        .find(|s| &s.model == model)
                        .and_then(|s| s.machines.iter().find(|(n, _)| n == &p.name))
                        .map(|(_, b)| *b)
                        .unwrap_or(0),
                })
                .collect();
            MachineView {
                id,
                name: p.name.clone(),
                chip: p.chip.clone(),
                backend: p.backend.label().into(),
                os: p.os.clone(),
                cpu_cores: p.cpu_cores,
                gpu_cores: p.gpu_cores,
                total_memory: p.accel_memory.unwrap_or(p.total_memory).0,
                usable_memory: p.usable_memory.0,
                available_memory: p.available_memory.map(|b| b.0),
                weight_gbs: p.measured_bandwidth_gbs,
                tflops: p.measured_tflops,
                spec_bandwidth_gbs: p.bandwidth_gbs,
                spec_tflops: p.tflops,
                local,
                address,
                rtt_ms,
                bandwidth_gbps: gbps,
                on_battery: p.on_battery,
                roles,
            }
        };
        let mut out = Vec::new();
        if let Some(p) = &self.local {
            out.push(view(
                machine_key(LOCAL_MACHINE),
                p,
                true,
                "this server".into(),
                None,
                None,
            ));
        }
        let lobby = self.lobby.lock().unwrap();
        let mut ids: Vec<&u64> = lobby.keys().collect();
        ids.sort_by_key(|id| lobby[id].profile.name.clone());
        for id in ids {
            let e = &lobby[id];
            out.push(view(
                machine_key(*id),
                &e.profile,
                false,
                e.peer.to_string(),
                e.rtt_ms,
                e.gbps,
            ));
        }
        out
    }

    /// Every machine of the pool as one cluster for planning, with each
    /// machine's id. Links are measured to this server; a hop between two
    /// joined machines is estimated from the slower of their links.
    fn pool_cluster(&self, link: Option<Link>) -> (Cluster, Vec<u64>) {
        let default = link.clone().unwrap_or_else(|| Link::preset("gbe").unwrap());
        let mut nodes = Vec::new();
        let mut ids = Vec::new();
        let mut links: Vec<Option<Link>> = Vec::new();
        // Speeds learned while serving refine the benchmarks.
        let cal = crate::calibration::Calibration::load();
        if let Some(p) = &self.local {
            let mut p = p.clone();
            cal.apply(&mut p);
            nodes.push(p);
            ids.push(LOCAL_MACHINE);
            links.push(Link::preset("local"));
        }
        let lobby = self.lobby.lock().unwrap();
        let mut order: Vec<&u64> = lobby.keys().collect();
        order.sort_by_key(|id| lobby[id].profile.name.clone());
        for id in order {
            let e = &lobby[id];
            let mut p = e.profile.clone();
            cal.apply(&mut p);
            nodes.push(p);
            ids.push(*id);
            links.push(match (e.rtt_ms, e.gbps) {
                (Some(rtt), Some(bw)) => Some(Link {
                    name: "measured".into(),
                    bandwidth_gbps: bw,
                    rtt_ms: rtt,
                    measured: true,
                }),
                _ => None,
            });
        }
        let mut c = Cluster::new(nodes, default);
        c.dedupe_names();
        if link.is_none() {
            let local = ids.first() == Some(&LOCAL_MACHINE);
            for i in 0..ids.len() {
                for j in i + 1..ids.len() {
                    let l = if local && i == 0 {
                        links[j].clone()
                    } else {
                        match (&links[i], &links[j]) {
                            (Some(x), Some(y)) => Some(Link {
                                name: "estimated".into(),
                                bandwidth_gbps: x.bandwidth_gbps.min(y.bandwidth_gbps),
                                rtt_ms: x.rtt_ms.max(y.rtt_ms),
                                measured: false,
                            }),
                            _ => None,
                        }
                    };
                    if let Some(l) = l {
                        let (a, b) = (c.nodes[i].name.clone(), c.nodes[j].name.clone());
                        c.set_link(&a, &b, l);
                    }
                }
            }
        }
        (c, ids)
    }

    /// Plan a model across the pool's machines (next to the models already
    /// loaded) without loading or downloading anything.
    pub async fn plan(&self, req: &LoadRequest) -> Result<PlanPreview> {
        let opts = self.resolve(req).await?;
        if self.names().contains(&opts.model_name) {
            bail!("{} is already loaded", opts.model_name);
        }
        let spec = model_spec(&opts)?;
        let (cluster, ids) = self.pool_cluster(opts.link.clone());
        // Loaded models keep their machines where they can.
        let loaded: Vec<(ModelSpec, Workload, PlanOptions, Vec<usize>)> = self
            .models()
            .iter()
            .map(|c| {
                let share = self
                    .shares()
                    .into_iter()
                    .find(|s| s.model == c.inner.opts.model_name);
                let current = cluster
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| {
                        share
                            .as_ref()
                            .is_some_and(|s| s.machines.iter().any(|(m, _)| m == &n.name))
                    })
                    .map(|(i, _)| i)
                    .collect();
                (
                    c.inner.spec.clone(),
                    c.workload(),
                    c.plan_options(),
                    current,
                )
            })
            .collect();
        let names: Vec<String> = self.names();
        let w = crate::coordinator::workload_for(&opts);
        let po = crate::coordinator::plan_options_for(&opts);
        // The same model in every weight format, to offer alternatives.
        let mut formats = Vec::new();
        for f in [
            WeightFormat::Native,
            WeightFormat::Q8_0,
            WeightFormat::Q6K,
            WeightFormat::Q4K,
        ] {
            if f != opts.format {
                let mut o = opts.clone();
                o.format = f;
                formats.push((f, model_spec(&o)?));
            }
        }
        let preview = tokio::task::spawn_blocking(move || {
            preview(&spec, &opts, w, po, &loaded, &names, &cluster, &formats)
        })
        .await?;
        let _ = ids;
        Ok(preview)
    }

    pub fn is_multi(&self) -> bool {
        self.models.lock().unwrap().len() > 1
    }

    /// The coordinator for an API request's `model` field. With a single
    /// model any name is accepted (clients often send their own label).
    pub fn get(&self, name: Option<&str>) -> Result<Coordinator, String> {
        let models = self.models();
        let Some(first) = models.first() else {
            return Err("no model is loaded yet — run `tendril load <model>` on the server".into());
        };
        let Some(name) = name.map(str::trim).filter(|n| !n.is_empty()) else {
            return Ok(first.clone());
        };
        if models.len() == 1 {
            return Ok(first.clone());
        }
        let names = self.names();
        let norm = |s: &str| s.to_ascii_lowercase();
        let want = norm(name);
        // Exact, then case-insensitive, then the part after the org ("Qwen/…"),
        // then a unique substring.
        let short = |s: &str| norm(s.rsplit('/').next().unwrap_or(s));
        let pick = names
            .iter()
            .position(|n| n == name)
            .or_else(|| names.iter().position(|n| norm(n) == want))
            .or_else(|| names.iter().position(|n| short(n) == short(&want)))
            .or_else(|| {
                let hits: Vec<usize> = (0..names.len())
                    .filter(|&i| norm(&names[i]).contains(&want))
                    .collect();
                (hits.len() == 1).then(|| hits[0])
            });
        match pick {
            Some(i) => Ok(models[i].clone()),
            None => Err(format!(
                "model '{name}' is not served here; available: {}",
                names.join(", ")
            )),
        }
    }

    pub fn shares(&self) -> Vec<Share> {
        self.shares.lock().unwrap().clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.events.subscribe()
    }

    /// Every model's events, oldest first.
    pub fn history(&self) -> Vec<Event> {
        self.history.lock().unwrap().iter().cloned().collect()
    }

    fn push_event(
        events: &broadcast::Sender<Event>,
        history: &StdMutex<VecDeque<Event>>,
        e: Event,
    ) {
        {
            let mut h = history.lock().unwrap();
            h.push_back(e.clone());
            while h.len() > 300 {
                h.pop_front();
            }
        }
        let _ = events.send(e);
    }

    fn event(&self, level: &'static str, text: impl Into<String>) {
        let e = Event {
            at: crate::coordinator::clock(),
            level,
            text: text.into(),
            model: None,
        };
        tracing::info!("{}", e.text);
        Self::push_event(&self.events, &self.history, e);
    }

    /// Merge a model's event stream into the pool's.
    fn forward_events(&self, c: &Coordinator) {
        // Events logged before we subscribed.
        for e in c.history() {
            Self::push_event(&self.events, &self.history, e);
        }
        let mut rx = c.subscribe();
        let tx = self.events.clone();
        let history = self.history.clone();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(e) => Self::push_event(&tx, &history, e),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                }
            }
        });
    }

    /// One short word for the whole pool (discovery, terminal).
    pub fn state(&self) -> &'static str {
        let st: Vec<Status> = self.models().iter().map(|c| c.status()).collect();
        if st.is_empty() {
            "waiting for a model"
        } else if st.iter().all(|s| matches!(s, Status::Ready { .. })) {
            "ready"
        } else if st.iter().any(|s| matches!(s, Status::Loading { .. })) {
            "loading"
        } else if st.iter().any(|s| matches!(s, Status::Failed { .. })) {
            "failed"
        } else if st.iter().any(|s| matches!(s, Status::Ready { .. })) {
            "partly ready"
        } else if st.iter().any(|s| matches!(s, Status::Waiting { .. })) {
            "waiting for machines"
        } else {
            "starting"
        }
    }

    /// Distinct machines in the pool.
    pub fn machine_count(&self) -> usize {
        // ponytail: counts this machine even with --no-local.
        let joined = self.lobby.lock().unwrap().len() + 1;
        self.models()
            .iter()
            .map(|c| c.nodes().len())
            .fold(joined, usize::max)
    }

    pub async fn shutdown(&self) {
        for c in self.models() {
            c.shutdown(true).await;
        }
        for e in self.lobby.lock().unwrap().values() {
            let _ = e.tx.send(Msg::Bye {
                reason: "the host stopped the cluster".into(),
                leave: true,
            });
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // ------------------------------------------------------------------

    async fn control_loop(self: Arc<Self>, listener: TcpListener, psk: [u8; 32]) {
        loop {
            let Ok((s, _)) = listener.accept().await else {
                continue;
            };
            let pool = self.clone();
            tokio::spawn(async move {
                if let Err(e) = pool.session(s, psk).await {
                    tracing::debug!("agent session ended: {e:#}");
                }
            });
        }
    }

    async fn session(&self, s: tokio::net::TcpStream, psk: [u8; 32]) -> Result<()> {
        let local_ip = s.local_addr()?.ip();
        let conn = match wire::accept(s, &psk).await {
            Ok(c) => c,
            Err(e) => {
                self.event("warn", format!("Rejected a connection: {e:#}"));
                return Err(e);
            }
        };
        let peer_ip = conn.peer.ip();
        let (mut reader, mut writer) = (conn.reader, conn.writer);
        let (hello, model) = read_hello(&mut reader, &mut writer).await?;
        let models = self.models();
        let coord = match &model {
            None => return self.lobby_session(reader, writer, hello, peer_ip).await,
            Some(m) => match models.iter().find(|c| &c.inner.opts.model_name == m) {
                Some(c) => c,
                None => {
                    let reason = format!("this cluster doesn't serve {m} any more");
                    writer
                        .send(&crate::proto::Msg::Reject {
                            reason: reason.clone(),
                        })
                        .await?;
                    bail!(reason);
                }
            },
        };
        coord
            .serve_agent(reader, writer, peer_ip, local_ip, hello)
            .await
    }

    /// A machine's first session: keep it alive and tell it about models.
    async fn lobby_session(
        &self,
        mut reader: wire::Reader,
        mut writer: wire::Writer,
        hello: crate::coordinator::HelloInfo,
        peer: IpAddr,
    ) -> Result<()> {
        let name = hello.profile.name.clone();
        writer
            .send(&Msg::Welcome {
                node_id: 0,
                name: name.clone(),
                cluster: String::new(),
                models: self.names(),
            })
            .await?;
        let (tx, mut rx) = mpsc::unbounded_channel::<Msg>();
        self.lobby.lock().unwrap().insert(
            hello.machine,
            LobbyEntry {
                profile: hello.profile.clone(),
                tx: tx.clone(),
                peer,
                rtt_ms: None,
                gbps: None,
            },
        );
        // Measure the link to this server, so plans made before any model
        // is loaded already know it: five pings one after another, then a
        // 4 MiB transfer.
        let _ = tx.send(Msg::Ping { t: 1 });
        let mut sent = Instant::now();
        let speed = match hello.profile.measured_bandwidth_gbs {
            Some(g) => format!(" · decode {g:.0} GB/s measured"),
            None => String::new(),
        };
        self.event(
            "ok",
            format!(
                "{name} joined — {} · {} · {} for models{speed}{}",
                hello.profile.chip,
                hello.profile.backend.label(),
                hello.profile.usable_memory,
                if self.models().is_empty() {
                    ". Pick a model in the dashboard or with `tendril load <model>`"
                } else {
                    ""
                }
            ),
        );
        let pump = tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                let m = tokio::select! {
                    m = rx.recv() => match m { Some(m) => m, None => break },
                    _ = tick.tick() => Msg::Ping { t: 0 },
                };
                if writer.send(&m).await.is_err() {
                    break;
                }
            }
        });
        // Pongs keep the session alive; silence means the machine is gone.
        let mut rtts = Vec::new();
        while let Ok(Ok(m)) = tokio::time::timeout(Duration::from_secs(20), reader.recv()).await {
            let mut lobby = self.lobby.lock().unwrap();
            let Some(e) = lobby.get_mut(&hello.machine) else {
                continue;
            };
            match m {
                Msg::Pong { t } if (1..=5).contains(&t) => {
                    rtts.push(sent.elapsed().as_secs_f64() * 1000.0);
                    let mut s = rtts.clone();
                    s.sort_by(f64::total_cmp);
                    e.rtt_ms = Some(s[s.len() / 2]);
                    let _ = tx.send(if t < 5 {
                        Msg::Ping { t: t + 1 }
                    } else {
                        Msg::Probe {
                            data: vec![0x5a; 4 << 20],
                        }
                    });
                    sent = Instant::now();
                }
                Msg::ProbeAck { len } => {
                    let rtt = e.rtt_ms.unwrap_or(0.0) / 1000.0;
                    let secs = (sent.elapsed().as_secs_f64() - rtt / 2.0).max(1e-4);
                    e.gbps = Some(len as f64 * 8.0 / secs / 1e9);
                }
                _ => {}
            }
        }
        pump.abort();
        let mut lobby = self.lobby.lock().unwrap();
        if lobby
            .get(&hello.machine)
            .is_some_and(|e| e.tx.same_channel(&tx))
        {
            lobby.remove(&hello.machine);
        }
        drop(lobby);
        self.event("warn", format!("{name} left the cluster"));
        Ok(())
    }

    async fn allocator_loop(self: Arc<Self>, changed: Arc<Notify>) {
        let mut last: Vec<HashMap<u64, Bytes>> = vec![];
        let mut first = true;
        loop {
            changed.notified().await;
            // Let a machine's sessions for every model arrive.
            tokio::time::sleep(Duration::from_millis(600)).await;
            let models = self.models();
            let links = self.links.lock().unwrap().clone();
            if models.is_empty() {
                continue;
            }
            // A model added since the last round starts with no share.
            let grew = last.len() < models.len();
            last.resize(models.len(), HashMap::new());
            // One view of the machines for everything the pool decides: the
            // plan preview, every model's share and the split it runs.
            let (cluster, machines) = self.pool_cluster(models[0].inner.opts.link.clone());
            let specs: Vec<_> = models
                .iter()
                .zip(&last)
                .map(|(c, prev)| {
                    let current: Vec<usize> = (0..machines.len())
                        .filter(|&j| prev.contains_key(&machines[j]))
                        .collect();
                    (
                        c.inner.spec.clone(),
                        c.workload(),
                        c.plan_options(),
                        current,
                    )
                })
                .collect();
            let cl = cluster.clone();
            let alloc = match tokio::task::spawn_blocking(move || {
                let pm: Vec<PoolModel> = specs
                    .iter()
                    .map(|(s, w, o, cur)| PoolModel {
                        spec: s,
                        workload: w.clone(),
                        opts: o.clone(),
                        current: cur.clone(),
                    })
                    .collect();
                allocate(&pm, &cl)
            })
            .await
            {
                Ok(a) => a,
                Err(_) => continue,
            };
            let names = self.names();
            let mut next: Vec<HashMap<u64, Bytes>> = Vec::new();
            let mut shares = Vec::new();
            for (i, p) in alloc.placements.iter().enumerate() {
                let mut map = HashMap::new();
                let mut list = Vec::new();
                for (j, b) in p.budgets.iter().enumerate() {
                    if b.0 > 0 {
                        map.insert(machines[j], *b);
                        list.push((cluster.nodes[j].name.clone(), b.0));
                    }
                }
                next.push(map);
                shares.push(Share {
                    model: names[i].clone(),
                    placed: p.plan.is_some(),
                    reason: p.reason.clone(),
                    machines: list,
                });
            }
            *self.shares.lock().unwrap() = shares.clone();
            if next == last && !first && !grew {
                continue;
            }
            first = false;
            if !cluster.nodes.is_empty() {
                let parts: Vec<String> = shares
                    .iter()
                    .map(|s| {
                        if s.placed {
                            format!(
                                "{} → {}",
                                s.model,
                                s.machines
                                    .iter()
                                    .map(|(n, b)| format!("{n} ({})", Bytes(*b)))
                                    .collect::<Vec<_>>()
                                    .join(" + ")
                            )
                        } else {
                            format!(
                                "{} waits ({})",
                                s.model,
                                s.reason.as_deref().unwrap_or("no room")
                            )
                        }
                    })
                    .collect();
                self.event(
                    if alloc.placed() == shares.len() {
                        "info"
                    } else {
                        "warn"
                    },
                    format!(
                        "Sharing {} machine(s): {}",
                        cluster.nodes.len(),
                        parts.join(" · ")
                    ),
                );
            }
            // Shrink first so memory is free before another model loads into it.
            let shrinks = |i: usize| {
                last[i]
                    .iter()
                    .any(|(m, b)| next[i].get(m).is_none_or(|nb| nb < b))
            };
            let (shrinking, rest): (Vec<usize>, Vec<usize>) =
                (0..models.len()).partition(|&i| shrinks(i));
            let apply = |i: usize| {
                let placed = alloc.placements[i].plan.is_some();
                *links[i].note.lock().unwrap() = if placed {
                    None
                } else {
                    Some(
                        alloc.placements[i]
                            .reason
                            .clone()
                            .unwrap_or_else(|| "not enough memory in the pool".into()),
                    )
                };
                *links[i].budgets.lock().unwrap() = Some(next[i].clone());
                *links[i].plan.lock().unwrap() = alloc.placements[i].plan.clone().map(|p| {
                    let ms = p.stages.iter().map(|st| machines[st.node]).collect();
                    (p, ms)
                });
                models[i].replan();
            };
            for &i in &shrinking {
                apply(i);
            }
            if !shrinking.is_empty() && !rest.is_empty() {
                tokio::time::sleep(Duration::from_millis(1500)).await;
            }
            for &i in &rest {
                apply(i);
            }
            last = next;
        }
    }
}

/// How the dashboard and control requests name a machine.
pub fn machine_key(machine: u64) -> String {
    format!("{machine:x}")
}

/// The plan preview: place the candidate next to the loaded models.
#[allow(clippy::too_many_arguments)]
fn preview(
    spec: &ModelSpec,
    opts: &ServeOptions,
    w: Workload,
    po: PlanOptions,
    loaded: &[(ModelSpec, Workload, PlanOptions, Vec<usize>)],
    names: &[String],
    cluster: &Cluster,
    formats: &[(WeightFormat, ModelSpec)],
) -> PlanPreview {
    let place = |cand: &ModelSpec| {
        let mut pm: Vec<PoolModel> = loaded
            .iter()
            .map(|(s, w, o, cur)| PoolModel {
                spec: s,
                workload: w.clone(),
                opts: o.clone(),
                current: cur.clone(),
            })
            .collect();
        pm.push(PoolModel {
            spec: cand,
            workload: w.clone(),
            opts: po.clone(),
            current: vec![],
        });
        allocate(&pm, cluster)
    };
    let machine_names: Vec<String> = cluster.nodes.iter().map(|n| n.name.clone()).collect();
    let mut out = PlanPreview {
        model: opts.model_name.clone(),
        architecture: spec.arch.label().to_string(),
        params: spec.total_params(),
        layers: spec.num_layers,
        format: opts.format.label().to_string(),
        weight_bytes: spec.weight_bytes().0,
        context: opts.context,
        concurrency: opts.concurrency,
        machines: cluster.nodes.len(),
        fits: false,
        reason: None,
        shares: vec![],
        tokens_per_sec: None,
        ttft_ms: None,
        plan: None,
        machine_names,
        displaces: vec![],
        alternatives: vec![],
        advice: vec![],
    };
    if cluster.nodes.is_empty() {
        out.reason = Some(
            "No machines in the pool yet — run `tendril join` on a machine (this server was started with --no-local)".into(),
        );
        return out;
    }
    let a = place(spec);
    let mine = a.placements.last().expect("candidate placed last");
    out.fits = mine.plan.is_some();
    out.reason = mine.reason.clone();
    out.displaces = a
        .placements
        .iter()
        .zip(names)
        .filter(|(p, _)| p.plan.is_none())
        .map(|(_, n)| n.clone())
        .collect();
    if let Some(p) = &mine.plan {
        let total: u64 = p.stages.iter().map(|s| s.mem.weights.0).sum::<u64>().max(1);
        out.tokens_per_sec = Some(p.tokens_per_sec);
        out.ttft_ms = Some(p.ttft_ms);
        out.shares = p
            .stages
            .iter()
            .map(|s| DeviceShare {
                machine: cluster.nodes[s.node].name.clone(),
                backend: s.backend.label().to_string(),
                layers: (s.layer_start, s.layer_end),
                components: s.describe_components(),
                weight_bytes: s.mem.weights.0,
                weight_pct: s.mem.weights.0 as f64 / total as f64 * 100.0,
                memory_needed: s.mem.peak.0,
                memory_budget: mine.budgets[s.node].0,
                kv_bytes: s.mem.kv.0,
                decode_ms: s.decode_ms,
            })
            .collect();
    } else {
        // Advice for the model on its own (what would make it fit).
        let alone = tendril_core::planner::plan(spec, cluster, &w, &po);
        out.advice = tendril_core::advice::advise(spec, cluster, &w, &po, &alone, "tendril load")
            .iter()
            .map(|a| format!("{} — {}", a.title, a.detail))
            .collect();
    }
    out.plan = mine.plan.clone();
    for (f, s) in formats {
        let a = place(s);
        let p = a.placements.last().expect("candidate placed last");
        out.alternatives.push(Alternative {
            format: f.label().to_string(),
            weight_bytes: s.weight_bytes().0,
            fits: p.plan.is_some(),
            tokens_per_sec: p.plan.as_ref().map(|p| p.tokens_per_sec),
        });
    }
    out
}
