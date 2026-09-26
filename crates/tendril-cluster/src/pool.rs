//! Several models served from one pool of machines.
//!
//! Each model runs its own coordinator (pipeline, batching, caches). The pool
//! owns the control port: every joining machine opens one session per model,
//! all tagged with the same machine id, and the pool splits each machine's
//! memory between the models with `tendril_core::pool::allocate`. A model's
//! coordinator then plans inside its share like it would on a smaller machine.

use crate::coordinator::{read_hello, Coordinator, Event, PoolLink, ServeOptions, Status};
use crate::proto::Msg;
use crate::{token, wire};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tendril_core::cluster::Cluster;
use tendril_core::hardware::NodeProfile;
use tendril_core::pool::{allocate, PoolModel};
use tendril_core::units::Bytes;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, mpsc, Notify};

/// Turns what a user typed into serve options (fetching the model's config).
pub type Resolver = Arc<dyn Fn(&str) -> Result<ServeOptions> + Send + Sync>;

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
    lobby: StdMutex<HashMap<u64, (NodeProfile, mpsc::UnboundedSender<Msg>)>>,
    adding: tokio::sync::Mutex<()>,
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
        let pool = Pool::open(first.control_port, first.token.clone(), None).await?;
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
        });
        tokio::spawn(pool.clone().control_loop(control, token::psk(&token)));
        tokio::spawn(pool.clone().allocator_loop(changed));
        Ok(pool)
    }

    /// Resolve `name` (downloading its config, not its weights) and serve it.
    pub async fn load(&self, name: &str) -> Result<String> {
        let resolve = self
            .resolve
            .clone()
            .context("this server can't load models on request")?;
        let name = name.to_string();
        let opts = tokio::task::spawn_blocking(move || resolve(&name)).await??;
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
            index: names.len() - 1,
            models: names.clone(),
            budgets: StdMutex::new(None),
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
        // Joined machines open a session for the new model.
        for (_, tx) in self.lobby.lock().unwrap().values() {
            let _ = tx.send(Msg::Models {
                models: names.clone(),
            });
        }
        self.changed.notify_one();
        Ok(())
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
            .map(|(p, _)| p.clone())
            .collect()
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
            c.shutdown().await;
        }
        for (_, tx) in self.lobby.lock().unwrap().values() {
            let _ = tx.send(Msg::Bye {
                reason: "coordinator shutting down".into(),
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
            None => return self.lobby_session(reader, writer, hello).await,
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
        self.lobby
            .lock()
            .unwrap()
            .insert(hello.machine, (hello.profile.clone(), tx.clone()));
        if self.models().is_empty() {
            self.event(
                "ok",
                format!(
                    "{name} joined — {} · {} for models. Waiting for a model: `tendril load <model>`",
                    hello.profile.chip, hello.profile.usable_memory
                ),
            );
        }
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
        while let Ok(Ok(_)) = tokio::time::timeout(Duration::from_secs(20), reader.recv()).await {}
        pump.abort();
        let mut lobby = self.lobby.lock().unwrap();
        if lobby
            .get(&hello.machine)
            .is_some_and(|(_, t)| t.same_channel(&tx))
        {
            lobby.remove(&hello.machine);
        }
        drop(lobby);
        if self.models().is_empty() {
            self.event("warn", format!("{name} left the pool"));
        }
        Ok(())
    }

    /// Every machine any model sees, with the best-known links between them.
    fn merged_cluster(&self, models: &[Coordinator]) -> (Cluster, Vec<u64>) {
        let snaps: Vec<(Cluster, Vec<u32>, Vec<u64>)> =
            models.iter().map(|c| c.raw_snapshot()).collect();
        let mut machines: Vec<u64> = Vec::new();
        let mut nodes: Vec<NodeProfile> = Vec::new();
        for (c, _, ms) in &snaps {
            for (i, m) in ms.iter().enumerate() {
                if !machines.contains(m) {
                    machines.push(*m);
                    nodes.push(c.nodes[i].clone());
                }
            }
        }
        let default = snaps[0].0.default_link.clone();
        let mut merged = Cluster::new(nodes, default);
        merged.dedupe_names();
        for a in 0..machines.len() {
            for b in a + 1..machines.len() {
                let link = snaps.iter().find_map(|(c, _, ms)| {
                    let ia = ms.iter().position(|m| *m == machines[a])?;
                    let ib = ms.iter().position(|m| *m == machines[b])?;
                    Some(c.link(ia, ib).clone())
                });
                if let Some(l) = link {
                    let (na, nb) = (merged.nodes[a].name.clone(), merged.nodes[b].name.clone());
                    merged.set_link(&na, &nb, l);
                }
            }
        }
        (merged, machines)
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
            let (cluster, machines) = self.merged_cluster(&models);
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
