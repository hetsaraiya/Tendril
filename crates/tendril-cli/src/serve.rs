//! `tendril serve` and `tendril join`: run a model across machines.

use crate::models::ensure_local;
use crate::ui::*;
use anyhow::{Context, Result};
use clap::Args;
use std::sync::Arc;
use std::time::Duration;
use tendril_cluster::agent::{AgentEvent, AgentOptions};
use tendril_cluster::coordinator::{Coordinator, ServeOptions, Status};
use tendril_cluster::http::{router, AppState};
use tendril_core::units::{parse_tokens, Bytes};
use tendril_core::{Goal, Link};
use tendril_engine::linear::WeightFormat;

pub const CONTROL_PORT: u16 = 7420;

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Model: HuggingFace id, catalog name or local folder.
    pub model: String,
    /// HTTP port for the web chat and the OpenAI-compatible API.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// Address to bind the HTTP server to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
    /// Maximum tokens per conversation.
    #[arg(long, short = 'c', default_value = "8k")]
    pub context: String,
    /// Conversations generated at the same time.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// Weight format: native, q8_0, q6_k, q4_k (applied only when you ask).
    #[arg(long, short = 'q', default_value = "native")]
    pub quantize: String,
    /// balanced, latency, throughput or memory.
    #[arg(long, short = 'g', default_value = "balanced")]
    pub goal: String,
    /// Port other machines join on.
    #[arg(long, default_value_t = CONTROL_PORT)]
    pub cluster_port: u16,
    /// Cluster token (default: generated once and saved in your config dir).
    #[arg(long, env = "TENDRIL_TOKEN")]
    pub token: Option<String>,
    /// Don't run any part of the model on this machine (coordinate only).
    #[arg(long)]
    pub no_local: bool,
    /// Use at least this many machines (e.g. to measure distribution cost).
    #[arg(long, default_value_t = 1)]
    pub min_machines: usize,
    /// Most memory Tendril may use on this machine, e.g. 8gb.
    #[arg(long)]
    pub max_memory: Option<String>,
    /// auto, cpu, metal or cuda for this machine's part.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Override measured links with a preset (thunderbolt, 10gbe, gbe, wifi…).
    #[arg(long)]
    pub link: Option<String>,
    /// Name this machine in the cluster.
    #[arg(long)]
    pub name: Option<String>,
    /// Seconds in-flight requests wait for the cluster to recover when a machine leaves.
    #[arg(long, default_value_t = 180, value_name = "SECONDS")]
    pub recovery_timeout: u64,
    /// Don't draft tokens from the conversation to speed up decoding.
    #[arg(long)]
    pub no_speculate: bool,
    /// Most tokens drafted per verification pass.
    #[arg(long, default_value_t = 6)]
    pub draft_tokens: usize,
    /// Don't keep finished conversations to speed up their next turn.
    #[arg(long)]
    pub no_prefix_cache: bool,
    /// Disk each machine may use for idle conversations' KV (0 = never spill).
    #[arg(long, default_value = "8gb")]
    pub kv_disk: String,
    #[arg(long)]
    pub offline: bool,
}

#[derive(Args, Debug)]
pub struct JoinArgs {
    /// Coordinator address (host or host:port). Omit to find it on the local network.
    pub address: Option<String>,
    /// Cluster token shown by `tendril serve`.
    #[arg(long, env = "TENDRIL_TOKEN")]
    pub token: String,
    /// Name for this machine.
    #[arg(long)]
    pub name: Option<String>,
    /// auto, cpu, metal or cuda.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Most memory Tendril may use on this machine, e.g. 8gb.
    #[arg(long)]
    pub max_memory: Option<String>,
    /// Port for activations from other machines (0 = any free port).
    #[arg(long, default_value_t = 0)]
    pub data_port: u16,
}

fn lan_ip() -> String {
    local_ip_address::local_ip()
        .map(|i| i.to_string())
        .unwrap_or_else(|_| "127.0.0.1".into())
}

fn level_mark(level: &str) -> String {
    match level {
        "ok" => ok_mark(),
        "warn" => warn_mark(),
        "error" => bad_mark(),
        _ => dim("·"),
    }
}

pub fn serve(a: ServeArgs) -> Result<()> {
    let format = WeightFormat::parse(&a.quantize)
        .ok_or_else(|| anyhow::anyhow!("unknown --quantize '{}'", a.quantize))?;
    let goal =
        Goal::parse(&a.goal).ok_or_else(|| anyhow::anyhow!("unknown --goal '{}'", a.goal))?;
    let context = parse_tokens(&a.context)
        .ok_or_else(|| anyhow::anyhow!("cannot parse --context '{}'", a.context))?
        as usize;
    let link = match &a.link {
        Some(l) => Some(
            Link::preset(l)
                .ok_or_else(|| anyhow::anyhow!("unknown --link '{l}' ({})", Link::names()))?,
        ),
        None => None,
    };
    let max_memory = match &a.max_memory {
        Some(m) => Some(
            Bytes::parse(m).ok_or_else(|| anyhow::anyhow!("cannot parse --max-memory '{m}'"))?,
        ),
        None => None,
    };
    let (dir, name) = ensure_local(&a.model, !a.offline)?;
    let token = match &a.token {
        Some(t) => t.clone(),
        None => tendril_cluster::token::load_or_create()?,
    };
    let opts = ServeOptions {
        model_dir: dir,
        model_name: name.clone(),
        format,
        context,
        concurrency: a.concurrency.max(1),
        goal,
        safety: 0.05,
        control_port: a.cluster_port,
        data_port: 0,
        token: token.clone(),
        device: a.device.clone(),
        use_local: !a.no_local,
        link,
        name: a.name.clone(),
        min_stages: a.min_machines.max(1),
        max_memory,
        prefix_cache: !a.no_prefix_cache,
        speculate: !a.no_speculate,
        draft_tokens: a.draft_tokens.clamp(1, 16),
        recovery_timeout: Duration::from_secs(a.recovery_timeout),
        kv_disk: Bytes::parse(&a.kv_disk)
            .ok_or_else(|| anyhow::anyhow!("cannot parse --kv-disk '{}'", a.kv_disk))?,
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let coord = Coordinator::start(opts).await?;
        let ip = lan_ip();
        let join = format!("tendril join --token {token}");
        let join_direct = format!("tendril join {ip}:{} --token {token}", a.cluster_port);
        // Let machines on the LAN find this cluster by its token fingerprint.
        {
            let c = coord.clone();
            let fp = tendril_cluster::discovery::fingerprint(&token);
            let model = name.clone();
            let host = ip.clone();
            let cport = a.cluster_port;
            tokio::spawn(async move {
                tendril_cluster::discovery::announce(tendril_cluster::discovery::DISCOVERY_PORT, move || {
                    tendril_cluster::discovery::Beacon {
                        app: "tendril".into(),
                        protocol: tendril_cluster::proto::PROTOCOL,
                        version: env!("CARGO_PKG_VERSION").into(),
                        model: model.clone(),
                        host: host.clone(),
                        control_port: cport,
                        fingerprint: fp.clone(),
                        machines: c.nodes().len(),
                        state: match c.status() {
                            Status::Ready { .. } => "ready",
                            Status::Loading { .. } => "loading",
                            Status::Waiting { .. } => "waiting for machines",
                            Status::Failed { .. } => "failed",
                            Status::Starting => "starting",
                        }
                        .into(),
                    }
                })
                .await
            });
        }
        let state = AppState { coord: coord.clone(), join_command: join.clone(), model_id: name.clone() };
        let app = router(state);
        let listener = tokio::net::TcpListener::bind((a.host.as_str(), a.port))
            .await
            .with_context(|| format!("port {} is busy — pick another with --port", a.port))?;
        let shown_host = if a.host == "0.0.0.0" { ip.clone() } else { a.host.clone() };
        println!();
        println!("{} {}", bold("Tendril · serving"), bold(cyan(&name)));
        println!("  {}  {}", dim("Web chat "), bold(format!("http://{shown_host}:{}", a.port)));
        println!("  {}  http://{shown_host}:{}/v1  {}", dim("API      "), a.port, dim("(OpenAI-compatible)"));
        println!("  {}  run this on another machine on your network to add it:", dim("Add more "));
        println!("             {}", cyan(&join));
        println!("             {}", dim(format!("(if your network blocks discovery: {join_direct})")));
        println!();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        // Mirror cluster events in the terminal.
        let mut events = coord.subscribe();
        for e in coord.history() {
            println!("{} {} {}", dim(&e.at), level_mark(e.level), e.text);
        }
        let c2 = coord.clone();
        let port = a.port;
        let printer = tokio::spawn(async move {
            let mut last_progress = String::new();
            let mut tick = tokio::time::interval(Duration::from_secs(2));
            let mut waiting_shown = false;
            loop {
                tokio::select! {
                    e = events.recv() => match e {
                        Ok(e) => {
                            println!("{} {} {}", dim(&e.at), level_mark(e.level), e.text);
                            if e.text.starts_with("Ready in") {
                                println!("{}", green(format!("         Chat at http://{shown_host}:{port} or run `tendril chat`")));
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                        Err(_) => break,
                    },
                    _ = tick.tick() => {
                        match c2.status() {
                            Status::Loading { stages, .. } => {
                                let line = stages.iter().filter(|s| s.phase != "ready").map(|s| {
                                    if s.total > 0 && s.phase == "sending" {
                                        format!("{} {} {}/{} ({:.0}%)", s.node, s.phase, Bytes(s.done), Bytes(s.total), s.done as f64 / s.total as f64 * 100.0)
                                    } else {
                                        format!("{} {}", s.node, s.phase)
                                    }
                                }).collect::<Vec<_>>().join(" · ");
                                if !line.is_empty() && line != last_progress {
                                    println!("{} {} {}", dim(chrono_now()), dim("…"), dim(&line));
                                    last_progress = line;
                                }
                            }
                            Status::Waiting { advice, .. } if !waiting_shown && !advice.is_empty() => {
                                waiting_shown = true;
                                for adv in advice.iter().take(3) {
                                    println!("         {} {}", cyan("→"), dim(adv));
                                }
                            }
                            Status::Ready { .. } => waiting_shown = false,
                            _ => {}
                        }
                    }
                }
            }
        });
        tokio::signal::ctrl_c().await.ok();
        println!();
        println!("{}", dim("Stopping… (telling joined machines to unload)"));
        coord.shutdown().await;
        printer.abort();
        Ok::<_, anyhow::Error>(())
    })
}

fn chrono_now() -> String {
    tendril_cluster::coordinator::clock()
}

pub fn join(a: JoinArgs) -> Result<()> {
    let address = match &a.address {
        Some(addr) if addr.contains(':') => addr.clone(),
        Some(addr) => format!("{addr}:{CONTROL_PORT}"),
        None => find_cluster(&a.token)?,
    };
    let max_memory = match &a.max_memory {
        Some(m) => Some(
            Bytes::parse(m).ok_or_else(|| anyhow::anyhow!("cannot parse --max-memory '{m}'"))?,
        ),
        None => None,
    };
    let me = tendril_core::hardware::detect_local(true);
    println!();
    println!("{} {}", bold("Tendril · joining"), bold(cyan(&address)));
    println!(
        "  {}  {} · {} backend · {} for models{}",
        dim("This machine"),
        me.chip,
        me.backend.label(),
        max_memory
            .map(|m| m.min(me.usable_memory))
            .unwrap_or(me.usable_memory),
        max_memory
            .map(|m| format!(" (capped at {m})"))
            .unwrap_or_default()
    );
    println!(
        "  {}",
        dim("Leave this running. Ctrl-C to leave the cluster.")
    );
    println!();
    let opts = AgentOptions {
        coordinator: address,
        token: a.token.clone(),
        name: a.name.clone(),
        device: a.device.clone(),
        data_port: a.data_port,
        cache: tendril_cluster::shard::default_cache(),
        once: false,
        max_memory,
    };
    let ev: tendril_cluster::agent::EventFn = Arc::new(|e: AgentEvent| {
        let t = tendril_cluster::coordinator::clock();
        let line = match e {
            AgentEvent::Connecting { addr } => format!("{} connecting to {addr}…", dim("·")),
            AgentEvent::Connected { name, cluster } => format!(
                "{} joined as {} — serving {}",
                ok_mark(),
                bold(name),
                bold(cluster)
            ),
            AgentEvent::Rejected { reason } => format!("{} {reason}", bad_mark()),
            AgentEvent::Disconnected { reason, retry_in } => {
                format!(
                    "{} lost the coordinator ({reason}); retrying in {}s",
                    warn_mark(),
                    retry_in.as_secs()
                )
            }
            AgentEvent::Receiving { spec, done, total } => format!(
                "{} receiving weights for {}: {}/{} ({:.0}%)",
                dim("↓"),
                tendril_cluster::coordinator::spec_label(&spec),
                Bytes(done),
                Bytes(total),
                done as f64 / total.max(1) as f64 * 100.0
            ),
            AgentEvent::Loading { spec, cached } => format!(
                "{} loading {}{}",
                dim("◌"),
                tendril_cluster::coordinator::spec_label(&spec),
                if cached {
                    dim(" (from cache)")
                } else {
                    String::new()
                }
            ),
            AgentEvent::Ready {
                spec,
                weight_bytes,
                load_ms,
                device,
            } => format!(
                "{} running {} ({}, {}) — ready in {:.1} s",
                ok_mark(),
                bold(tendril_cluster::coordinator::spec_label(&spec)),
                Bytes(weight_bytes),
                device,
                load_ms as f64 / 1000.0
            ),
            AgentEvent::Unloaded => format!("{} unloaded", dim("·")),
            AgentEvent::Failed { error } => format!("{} {error}", bad_mark()),
        };
        println!("{} {line}", dim(t));
    });
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        tokio::select! {
            r = tendril_cluster::agent::run(opts, ev) => r,
            _ = tokio::signal::ctrl_c() => {
                println!("\n{}", dim("Leaving the cluster."));
                Ok(())
            }
        }
    })
}

/// Start an in-process coordinator + HTTP server on a free port (for `bench <model>`).
pub async fn start_local_for_bench(
    model: &str,
    quantize: &str,
    concurrency: usize,
) -> Result<(String, Coordinator)> {
    let format = WeightFormat::parse(quantize)
        .ok_or_else(|| anyhow::anyhow!("unknown --quantize '{quantize}'"))?;
    let (dir, name) = ensure_local(model, true)?;
    let control = std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port();
    let opts = ServeOptions {
        model_dir: dir,
        model_name: name.clone(),
        format,
        context: 8192,
        concurrency: concurrency.max(1),
        goal: Goal::Balanced,
        safety: 0.05,
        control_port: control,
        data_port: 0,
        token: tendril_cluster::token::generate(),
        device: "auto".into(),
        use_local: true,
        link: None,
        name: None,
        min_stages: 1,
        max_memory: None,
        prefix_cache: true,
        kv_disk: Bytes::gib(1.0),
        speculate: true,
        draft_tokens: 6,
        recovery_timeout: Duration::from_secs(60),
    };
    let coord = Coordinator::start(opts).await?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let app = router(AppState {
        coord: coord.clone(),
        join_command: String::new(),
        model_id: name,
    });
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    eprint!("{} loading the model in-process… ", dim("·"));
    for _ in 0..6000 {
        match coord.status() {
            Status::Ready { .. } => {
                eprintln!("{}", ok_mark());
                return Ok((format!("http://127.0.0.1:{port}"), coord));
            }
            Status::Waiting { reason, .. } => {
                anyhow::bail!("the model does not fit on this machine: {reason}")
            }
            Status::Failed { error } => anyhow::bail!(error),
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
    anyhow::bail!("timed out loading the model")
}

/// Find the cluster whose token matches on the local network.
fn find_cluster(token: &str) -> Result<String> {
    use tendril_cluster::discovery::{discover, fingerprint, DISCOVERY_PORT};
    let fp = fingerprint(token);
    eprint!(
        "{} Looking for your cluster on the local network… ",
        dim("·")
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    for attempt in 0..10 {
        let found = rt.block_on(discover(DISCOVERY_PORT, Duration::from_secs(3)))?;
        if let Some((addr, b)) = found.iter().find(|(_, b)| b.fingerprint == fp) {
            eprintln!(
                "{} found {} serving {} ({})",
                ok_mark(),
                addr.ip(),
                bold(&b.model),
                b.state
            );
            return Ok(addr.to_string());
        }
        if attempt == 0 {
            if found.is_empty() {
                eprintln!(
                    "{}",
                    dim("nothing yet — is `tendril serve` running on this network? Still looking…")
                );
            } else {
                eprintln!(
                    "{}",
                    dim(format!(
                        "{} cluster(s) found, none with this token (check --token). Still looking…",
                        found.len()
                    ))
                );
            }
        }
    }
    anyhow::bail!(
        "no Tendril cluster with this token answered on the local network.\n  If the coordinator is on another subnet or your network blocks broadcast, pass its address:\n  tendril join <coordinator-ip> --token {token}"
    )
}

#[derive(Args, Debug)]
pub struct DiscoverArgs {
    /// Seconds to listen for clusters.
    #[arg(long, default_value_t = 3)]
    pub wait: u64,
}

/// `tendril discover`: list Tendril clusters on the local network.
pub fn discover_cmd(a: DiscoverArgs) -> Result<()> {
    use tendril_cluster::discovery::{discover, fingerprint, DISCOVERY_PORT};
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    eprintln!(
        "{} Listening for Tendril clusters for {} s…",
        dim("·"),
        a.wait
    );
    let found = rt.block_on(discover(DISCOVERY_PORT, Duration::from_secs(a.wait)))?;
    if found.is_empty() {
        println!("{} No clusters answered. Start one with `tendril serve <model>` (discovery needs broadcast on the LAN).", warn_mark());
        return Ok(());
    }
    let mine = tendril_cluster::token::load_or_create()
        .ok()
        .map(|t| fingerprint(&t));
    let mut t = crate::ui::Table::new(&["ADDRESS", "MODEL", "MACHINES", "STATE", ""]);
    for (addr, b) in &found {
        t.row(vec![
            bold(addr.to_string()),
            b.model.clone(),
            b.machines.to_string(),
            b.state.clone(),
            if mine.as_deref() == Some(b.fingerprint.as_str()) {
                green("your saved token")
            } else {
                String::new()
            },
        ]);
    }
    println!();
    t.print();
    println!();
    println!("{}", dim("Join one with: tendril join --token <its token>   (the token is printed by `tendril serve`)"));
    Ok(())
}
