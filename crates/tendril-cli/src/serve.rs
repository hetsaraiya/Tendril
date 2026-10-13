//! `tendril serve` and `tendril join`: a cluster of machines running
//! inference engines.

use crate::ui::*;
use anyhow::{Context, Result};
use clap::Args;
use std::sync::Arc;
use std::time::Duration;
use tendril_cluster::agent::{AgentEvent, AgentOptions};
use tendril_cluster::http::{router, AppState};
use tendril_cluster::pool::{LoadRequest, Pool, PoolOptions};
use tendril_cluster::proto::RoleState;
use tendril_core::units::{parse_tokens, Bytes};

pub const CONTROL_PORT: u16 = 7420;

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Models to load right away (with --engine). Give none to let machines
    /// join first, then pick in the dashboard or with `tendril load`.
    #[arg(value_name = "MODEL")]
    pub models: Vec<String>,
    /// Engine for the models given here (`tendril engines` lists them).
    #[arg(long, short = 'e', default_value = "llama.cpp")]
    pub engine: String,
    /// HTTP port for the dashboard and the OpenAI-compatible API.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// Address to bind the HTTP server to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
    /// Default tokens per conversation.
    #[arg(long, short = 'c', default_value = "8k")]
    pub context: String,
    /// Default conversations at the same time.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// Quantization for the models given here (engine-specific, e.g. Q4_K_M).
    #[arg(long, short = 'q')]
    pub quantize: Option<String>,
    /// Port other machines join on.
    #[arg(long, default_value_t = CONTROL_PORT)]
    pub cluster_port: u16,
    /// Port other machines reach this one's engines on (0 = any).
    #[arg(long, default_value_t = 0)]
    pub tunnel_port: u16,
    /// Cluster token (default: generated once and saved in your config dir).
    #[arg(long, env = "TENDRIL_TOKEN")]
    pub token: Option<String>,
    /// Don't run models on this machine (coordinate only).
    #[arg(long)]
    pub no_local: bool,
    /// Most memory models may use on this machine, e.g. 8gb.
    #[arg(long)]
    pub max_memory: Option<String>,
    /// Name this machine in the cluster.
    #[arg(long)]
    pub name: Option<String>,
    /// The machines share a private network you trust: allow engines whose
    /// traffic between machines isn't encrypted (vLLM, SGLang, exo) to span them.
    #[arg(long)]
    pub trusted_network: bool,
}

#[derive(Args, Debug)]
pub struct JoinArgs {
    /// Server address (host or host:port). Omit to find it on the local network.
    pub address: Option<String>,
    /// Cluster token shown by `tendril serve`.
    #[arg(long, env = "TENDRIL_TOKEN")]
    pub token: String,
    /// Name for this machine.
    #[arg(long)]
    pub name: Option<String>,
    /// Most memory models may use on this machine, e.g. 8gb.
    #[arg(long)]
    pub max_memory: Option<String>,
    /// Port other machines reach this one's engines on (0 = any free port;
    /// set it when a firewall needs a fixed port).
    #[arg(long, default_value_t = 0)]
    pub tunnel_port: u16,
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

/// Measure this machine once per hardware setup (CPU speed for the planner).
fn benchmark() {
    let p = tendril_cluster::machine::this_machine(None, None);
    if tendril_core::hardware::saved_benchmark(&p).is_some() {
        return;
    }
    eprint!(
        "{} Measuring this machine (first run on this hardware, a few seconds)… ",
        dim("·")
    );
    let mut b = tendril_core::hardware::measure(&p);
    b.measured_at = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();
    tendril_core::hardware::save_benchmark(&b);
    eprintln!(
        "{} memory {:.0} GB/s · CPU {:.2} TFLOP/s",
        ok_mark(),
        b.weight_gbs,
        b.tflops
    );
}

/// Print what an agent does, for `serve` (this machine) and `join`.
fn agent_printer(prefix: &'static str) -> tendril_cluster::agent::EventFn {
    let last: Arc<std::sync::Mutex<std::collections::HashMap<String, u64>>> = Default::default();
    Arc::new(move |e: AgentEvent| {
        let t = tendril_cluster::machine::clock();
        match e {
            AgentEvent::Connecting { addr } if prefix.is_empty() => {
                println!("{} {} connecting to {addr}…", dim(&t), dim("·"))
            }
            AgentEvent::Connected { name, cluster } if prefix.is_empty() => println!(
                "{} {} joined {} as {}",
                dim(&t),
                ok_mark(),
                bold(&cluster),
                bold(&name)
            ),
            AgentEvent::Rejected { reason } => println!("{} {} {reason}", dim(&t), bad_mark()),
            AgentEvent::Disconnected { reason, retry_in } => println!(
                "{} {} {prefix}disconnected ({reason}); retrying in {} s",
                dim(&t),
                warn_mark(),
                retry_in.as_secs()
            ),
            AgentEvent::Engines { found } if prefix.is_empty() => {
                let ready: Vec<&str> = found
                    .iter()
                    .filter(|(_, r)| *r)
                    .map(|(n, _)| n.as_str())
                    .collect();
                println!(
                    "{} {} engines here: {}",
                    dim(&t),
                    dim("·"),
                    if ready.is_empty() {
                        yellow("none installed — `tendril engines` shows how to add one")
                    } else {
                        bold(ready.join(", "))
                    }
                );
            }
            AgentEvent::Role {
                deployment,
                role,
                engine,
                state,
            } => {
                let label = format!("{prefix}{engine} {role} ({deployment})");
                match state {
                    RoleState::Downloading { file, done, total } => {
                        let pct = (done * 100).checked_div(total).unwrap_or(0);
                        let key = format!("{deployment}/{role}/{file}");
                        let mut l = last.lock().unwrap();
                        let prev = l.get(&key).copied();
                        if prev.is_none()
                            || pct >= prev.unwrap() + 20
                            || (done == total && prev != Some(100))
                        {
                            l.insert(key, pct);
                            println!(
                                "{} {} {label}: downloading {file} {}/{} ({pct}%)",
                                dim(&t),
                                dim("…"),
                                Bytes(done),
                                Bytes(total)
                            );
                        }
                    }
                    RoleState::Starting => println!("{} {} {label}: starting", dim(&t), dim("·")),
                    RoleState::Preparing { what, .. } => {
                        println!("{} {} {label}: {what}", dim(&t), dim("…"))
                    }
                    RoleState::Ready => println!("{} {} {label}: ready", dim(&t), ok_mark()),
                    RoleState::Failed { error, log } => {
                        println!("{} {} {label}: {error}", dim(&t), bad_mark());
                        for l in log.iter().rev().take(8).rev() {
                            println!("         {}", dim(l));
                        }
                    }
                    RoleState::Exited { code, log } => {
                        println!(
                            "{} {} {label}: exited ({})",
                            dim(&t),
                            warn_mark(),
                            code.map(|c| c.to_string())
                                .unwrap_or_else(|| "signal".into())
                        );
                        for l in log.iter().rev().take(8).rev() {
                            println!("         {}", dim(l));
                        }
                    }
                    RoleState::Started { .. } => {}
                }
            }
            AgentEvent::Left { reason } => println!("{} {} {reason}", dim(&t), dim("·")),
            _ => {}
        }
    })
}

pub fn serve(a: ServeArgs) -> Result<()> {
    let context = parse_tokens(&a.context)
        .ok_or_else(|| anyhow::anyhow!("cannot parse --context '{}'", a.context))?
        as usize;
    let max_memory = match &a.max_memory {
        Some(m) => Some(
            Bytes::parse(m).ok_or_else(|| anyhow::anyhow!("cannot parse --max-memory '{m}'"))?,
        ),
        None => None,
    };
    if !a.models.is_empty() && tendril_engines::get(&a.engine).is_none() {
        anyhow::bail!("unknown --engine '{}' (see `tendril engines`)", a.engine);
    }
    let token = match &a.token {
        Some(t) => t.clone(),
        None => tendril_cluster::token::load_or_create()?,
    };
    if !a.no_local {
        benchmark();
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let name = a
            .name
            .clone()
            .unwrap_or_else(|| tendril_cluster::machine::this_machine(None, None).name);
        let pool = Pool::start(PoolOptions {
            control_port: a.cluster_port,
            token: token.clone(),
            name: name.clone(),
            default_context: context,
            default_concurrency: a.concurrency.max(1),
        })
        .await?;
        if !a.no_local {
            let opts = AgentOptions {
                server: format!("127.0.0.1:{}", pool.control_port),
                token: token.clone(),
                name: a.name.clone(),
                tunnel_port: a.tunnel_port,
                cache: tendril_cluster::machine::default_cache(),
                once: false,
                max_memory,
                local: true,
            };
            let ev = agent_printer("this machine: ");
            tokio::spawn(async move {
                if let Err(e) = tendril_cluster::agent::run(opts, ev).await {
                    eprintln!("{} this machine's agent stopped: {e:#}", bad_mark());
                }
            });
        }
        let ip = lan_ip();
        let join = format!("tendril join --token {token}");
        let join_direct = format!("tendril join {ip}:{} --token {token}", pool.control_port);
        {
            let p = pool.clone();
            let fp = tendril_cluster::discovery::fingerprint(&token);
            let host = ip.clone();
            let cport = pool.control_port;
            tokio::spawn(async move {
                tendril_cluster::discovery::announce(tendril_cluster::discovery::DISCOVERY_PORT, move || {
                    tendril_cluster::discovery::Beacon {
                        app: "tendril".into(),
                        protocol: tendril_cluster::proto::PROTOCOL,
                        version: env!("CARGO_PKG_VERSION").into(),
                        model: p.deployments().iter().map(|d| d.name.clone()).collect::<Vec<_>>().join(", "),
                        host: host.clone(),
                        control_port: cport,
                        fingerprint: fp.clone(),
                        machines: p.machine_count(),
                        state: p.state_word().into(),
                    }
                })
                .await
            });
        }
        let app = router(AppState::new(pool.clone(), join.clone()));
        let listener = tokio::net::TcpListener::bind((a.host.as_str(), a.port))
            .await
            .with_context(|| format!("port {} is busy — pick another with --port", a.port))?;
        let shown_host = if a.host == "0.0.0.0" { ip.clone() } else { a.host.clone() };
        println!();
        println!("{}", bold(format!("Tendril · cluster {name}")));
        println!("  {}  {}  {}", dim("Dashboard"), bold(format!("http://{shown_host}:{}", a.port)), dim("(machines, engines, models, chat; controls unlock with the token)"));
        println!("  {}  http://{shown_host}:{}/v1  {}", dim("API      "), a.port, dim("(OpenAI-compatible, every engine)"));
        println!("  {}  run this on another machine on your network to add it:", dim("Add more "));
        println!("             {}", cyan(&join));
        println!("             {}", dim(format!("(if your network blocks discovery: {join_direct})")));
        println!("  {}  {}", dim("Token    "), bold(&token));
        println!();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        // Mirror cluster events in the terminal.
        let mut events = pool.subscribe();
        let port = a.port;
        let printer = tokio::spawn(async move {
            loop {
                match events.recv().await {
                    Ok(e) => {
                        let tag = e.model.as_deref().map(|m| format!("{} ", cyan(format!("[{}]", short_name(m))))).unwrap_or_default();
                        println!("{} {} {tag}{}", dim(&e.at), level_mark(e.level), e.text);
                        if e.level == "ok" && e.text.contains("is ready") {
                            println!("{}", green(format!("         Chat at http://{shown_host}:{port} or run `tendril chat`")));
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(_) => break,
                }
            }
        });
        if !a.models.is_empty() {
            // Give this machine's agent a moment to report its engines.
            let t0 = std::time::Instant::now();
            while pool.machine_count() == 0 && t0.elapsed() < Duration::from_secs(30) {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            for m in &a.models {
                let req = LoadRequest {
                    engine: a.engine.clone(),
                    model: m.clone(),
                    quantize: a.quantize.clone(),
                    trusted_network: a.trusted_network,
                    ..Default::default()
                };
                if let Err(e) = pool.load(&req).await {
                    println!("{} {m}: {e:#}", bad_mark());
                }
            }
        }
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = pool.stopped() => println!("\n{}", dim("Stop requested from the dashboard.")),
        }
        println!();
        println!("{}", dim("Stopping… (engines stop and joined machines exit)"));
        pool.shutdown().await;
        printer.abort();
        tokio::time::sleep(Duration::from_millis(500)).await;
        Ok::<_, anyhow::Error>(())
    })
}

/// "Qwen/Qwen2.5-7B-Instruct" → "Qwen2.5-7B-Instruct".
fn short_name(m: &str) -> &str {
    m.rsplit('/').next().unwrap_or(m)
}

pub fn join(a: JoinArgs) -> Result<()> {
    let max_memory = match &a.max_memory {
        Some(m) => Some(
            Bytes::parse(m).ok_or_else(|| anyhow::anyhow!("cannot parse --max-memory '{m}'"))?,
        ),
        None => None,
    };
    let server = match &a.address {
        Some(addr) if addr.contains(':') => addr.clone(),
        Some(addr) => format!("{addr}:{CONTROL_PORT}"),
        None => find_cluster(&a.token)?,
    };
    benchmark();
    let me = tendril_cluster::machine::this_machine(a.name.as_deref(), max_memory);
    println!();
    println!("{} {}", bold("Tendril · joining as"), bold(cyan(&me.name)));
    println!(
        "  {}  {} · {} usable",
        dim("Machine"),
        me.chip,
        me.usable_memory
    );
    println!(
        "  {}  engines run here as the server asks; Ctrl+C leaves the cluster",
        dim("Next   ")
    );
    println!();
    let opts = AgentOptions {
        server,
        token: a.token.clone(),
        name: a.name.clone(),
        tunnel_port: a.tunnel_port,
        cache: tendril_cluster::machine::default_cache(),
        once: false,
        max_memory,
        local: false,
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        tokio::select! {
            r = tendril_cluster::agent::run(opts, agent_printer("")) => r,
            _ = tokio::signal::ctrl_c() => {
                println!("\n{}", dim("Leaving the cluster."));
                Ok(())
            }
        }
    })
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
        "no Tendril cluster with this token answered on the local network.\n  If the server is on another subnet or your network blocks broadcast, pass its address:\n  tendril join <server-ip> --token {token}"
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
