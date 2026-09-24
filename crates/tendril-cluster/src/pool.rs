//! Several models served from one pool of machines.
//!
//! Each model runs its own coordinator (pipeline, batching, caches). The pool
//! owns the control port: every joining machine opens one session per model,
//! all tagged with the same machine id, and the pool splits each machine's
//! memory between the models with `tendril_core::pool::allocate`. A model's
//! coordinator then plans inside its share like it would on a smaller machine.

use crate::coordinator::{read_hello, Coordinator, Event, PoolLink, ServeOptions, Status};
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
use tokio::sync::{broadcast, Notify};

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
    models: Vec<Coordinator>,
    links: Vec<Arc<PoolLink>>,
    shares: StdMutex<Vec<Share>>,
    events: broadcast::Sender<Event>,
    history: Arc<StdMutex<VecDeque<Event>>>,
}

impl Pool {
    /// Start serving `opts` (one entry per model; the first one's ports,
    /// token and machine settings apply to the pool).
    pub async fn start(opts: Vec<ServeOptions>) -> Result<Arc<Pool>> {
        if opts.is_empty() {
            bail!("no model to serve");
        }
        let names: Vec<String> = opts.iter().map(|o| o.model_name.clone()).collect();
        for (i, n) in names.iter().enumerate() {
            if names[..i].contains(n) {
                bail!("{n} is listed twice");
            }
        }
        let (events, _) = broadcast::channel(512);
        let history = Arc::new(StdMutex::new(VecDeque::new()));
        if opts.len() == 1 {
            let c = Coordinator::start(opts.into_iter().next().unwrap()).await?;
            let pool = Arc::new(Pool {
                models: vec![c],
                links: vec![],
                shares: StdMutex::new(vec![]),
                events,
                history,
            });
            pool.forward_events();
            return Ok(pool);
        }
        let changed = Arc::new(Notify::new());
        let control_port = opts[0].control_port;
        let psk = token::psk(&opts[0].token);
        let control = TcpListener::bind(("0.0.0.0", control_port))
            .await
            .with_context(|| {
                format!("cannot listen on port {control_port} (is another Tendril running?)")
            })?;
        let mut models = Vec::new();
        let mut links = Vec::new();
        for (i, o) in opts.into_iter().enumerate() {
            let link = Arc::new(PoolLink {
                index: i,
                models: names.clone(),
                budgets: StdMutex::new(None),
                note: StdMutex::new(None),
                changed: changed.clone(),
            });
            models.push(Coordinator::start_with(o, Some(link.clone())).await?);
            links.push(link);
        }
        let pool = Arc::new(Pool {
            shares: StdMutex::new(
                names
                    .iter()
                    .map(|n| Share {
                        model: n.clone(),
                        placed: false,
                        reason: None,
                        machines: vec![],
                    })
                    .collect(),
            ),
            models,
            links,
            events,
            history,
        });
        pool.forward_events();
        tokio::spawn(pool.clone().control_loop(control, psk));
        tokio::spawn(pool.clone().allocator_loop(changed.clone()));
        changed.notify_one();
        Ok(pool)
    }

    pub fn primary(&self) -> &Coordinator {
        &self.models[0]
    }

    pub fn models(&self) -> &[Coordinator] {
        &self.models
    }

    pub fn names(&self) -> Vec<String> {
        self.models
            .iter()
            .map(|c| c.inner.opts.model_name.clone())
            .collect()
    }

    pub fn is_multi(&self) -> bool {
        self.models.len() > 1
    }

    /// The coordinator for an API request's `model` field. With a single
    /// model any name is accepted (clients often send their own label).
    pub fn get(&self, name: Option<&str>) -> Result<&Coordinator, String> {
        let Some(name) = name.map(str::trim).filter(|n| !n.is_empty()) else {
            return Ok(self.primary());
        };
        if !self.is_multi() {
            return Ok(self.primary());
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
            Some(i) => Ok(&self.models[i]),
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

    /// Merge the models' event streams into the pool's.
    fn forward_events(&self) {
        for c in &self.models {
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
    }

    /// One short word for the whole pool (discovery, terminal).
    pub fn state(&self) -> &'static str {
        let st: Vec<Status> = self.models.iter().map(|c| c.status()).collect();
        if st.iter().all(|s| matches!(s, Status::Ready { .. })) {
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
        self.models
            .iter()
            .map(|c| c.nodes().len())
            .max()
            .unwrap_or(0)
    }

    pub async fn shutdown(&self) {
        for c in &self.models {
            c.shutdown().await;
        }
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
                self.primary()
                    .event("warn", format!("Rejected a connection: {e:#}"));
                return Err(e);
            }
        };
        let peer_ip = conn.peer.ip();
        let (mut reader, mut writer) = (conn.reader, conn.writer);
        let (hello, model) = read_hello(&mut reader, &mut writer).await?;
        let coord = match &model {
            None => self.primary(),
            Some(m) => match self.models.iter().find(|c| &c.inner.opts.model_name == m) {
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

    /// Every machine any model sees, with the best-known links between them.
    fn merged_cluster(&self) -> (Cluster, Vec<u64>) {
        let snaps: Vec<(Cluster, Vec<u32>, Vec<u64>)> =
            self.models.iter().map(|c| c.raw_snapshot()).collect();
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
        let mut last: Vec<HashMap<u64, Bytes>> = vec![HashMap::new(); self.models.len()];
        let mut first = true;
        loop {
            changed.notified().await;
            // Let a machine's sessions for every model arrive.
            tokio::time::sleep(Duration::from_millis(600)).await;
            let (cluster, machines) = self.merged_cluster();
            let specs: Vec<_> = self
                .models
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
            if next == last && !first {
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
                (0..self.models.len()).partition(|&i| shrinks(i));
            let apply = |i: usize| {
                let placed = alloc.placements[i].plan.is_some();
                *self.links[i].note.lock().unwrap() = if placed {
                    None
                } else {
                    Some(
                        alloc.placements[i]
                            .reason
                            .clone()
                            .unwrap_or_else(|| "not enough memory in the pool".into()),
                    )
                };
                *self.links[i].budgets.lock().unwrap() = Some(next[i].clone());
                self.models[i].replan();
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
