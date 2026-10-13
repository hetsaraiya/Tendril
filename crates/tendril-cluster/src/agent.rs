//! The agent on every machine (`tendril join`, and inside `tendril serve`
//! for the server's own compute): reports the machine and its engines, runs
//! the engine processes the server asks for, and carries their traffic over
//! encrypted tunnels.

use crate::proto::{Msg, Peer, RoleState, PROTOCOL};
use crate::tunnel::{Forwarder, Route, Targets};
use crate::{fetch, token, wire};
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tendril_engines::{Detection, Launch, Ready, Role};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Clone, Debug)]
pub struct AgentOptions {
    /// The server's control address (`host:port`).
    pub server: String,
    pub token: String,
    pub name: Option<String>,
    /// Port other machines reach this one's engines on (0 = any).
    pub tunnel_port: u16,
    pub cache: PathBuf,
    /// Stop instead of reconnecting when the server goes away.
    pub once: bool,
    /// Cap on memory models may use here.
    pub max_memory: Option<tendril_core::Bytes>,
    /// This agent runs inside the server process.
    pub local: bool,
}

#[derive(Clone, Debug)]
pub enum AgentEvent {
    Connecting {
        addr: String,
    },
    Connected {
        name: String,
        cluster: String,
    },
    Rejected {
        reason: String,
    },
    Disconnected {
        reason: String,
        retry_in: Duration,
    },
    /// Engines found here: (name, ready).
    Engines {
        found: Vec<(String, bool)>,
    },
    Role {
        deployment: String,
        role: String,
        engine: String,
        state: RoleState,
    },
    /// The server stopped the cluster or removed this machine.
    Left {
        reason: String,
    },
}

pub type EventFn = Arc<dyn Fn(AgentEvent) + Send + Sync>;

/// A role this machine runs.
struct Running {
    task: JoinHandle<()>,
    /// Process group of the engine process, to stop it and its children.
    pid: Arc<Mutex<Option<u32>>>,
}

struct Agent {
    opts: AgentOptions,
    psk: [u8; 32],
    ev: EventFn,
    targets: Targets,
    tunnel_port: u16,
    roles: Mutex<HashMap<String, Running>>,
    engines: tokio::sync::Mutex<Option<Vec<Detection>>>,
}

enum End {
    Rejected(String),
    Bye { reason: String, leave: bool },
}

pub async fn run(opts: AgentOptions, ev: EventFn) -> Result<()> {
    let psk = token::psk(&opts.token);
    let listener = TcpListener::bind(("0.0.0.0", opts.tunnel_port))
        .await
        .with_context(|| format!("cannot listen on tunnel port {}", opts.tunnel_port))?;
    let tunnel_port = listener.local_addr()?.port();
    let targets = Targets::default();
    tokio::spawn(crate::tunnel::serve(listener, psk, targets.clone()));
    let agent = Arc::new(Agent {
        opts,
        psk,
        ev,
        targets,
        tunnel_port,
        roles: Mutex::new(HashMap::new()),
        engines: tokio::sync::Mutex::new(None),
    });
    let machine = rand::random::<u64>() | 1;
    let mut backoff = Duration::from_secs(1);
    let ev = agent.ev.clone();
    loop {
        ev(AgentEvent::Connecting {
            addr: agent.opts.server.clone(),
        });
        let started = Instant::now();
        let r = agent.clone().session(machine).await;
        agent.stop_all();
        match r {
            Ok(End::Rejected(reason)) => {
                ev(AgentEvent::Rejected {
                    reason: reason.clone(),
                });
                if agent.opts.once {
                    bail!(reason);
                }
            }
            Ok(End::Bye {
                reason,
                leave: true,
            }) => {
                ev(AgentEvent::Left { reason });
                return Ok(());
            }
            Ok(End::Bye { .. }) if agent.opts.once => return Ok(()),
            Ok(End::Bye { .. }) => {}
            Err(e) => {
                if started.elapsed() > Duration::from_secs(30) {
                    backoff = Duration::from_secs(1);
                }
                let msg = format!("{e:#}");
                if msg.contains("wrong cluster token") || msg.contains("authentication") {
                    ev(AgentEvent::Rejected {
                        reason: "the server rejected our token — check `--token`".into(),
                    });
                    bail!("wrong cluster token");
                }
                if agent.opts.once {
                    return Err(e);
                }
                ev(AgentEvent::Disconnected {
                    reason: msg,
                    retry_in: backoff,
                });
            }
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(10));
    }
}

impl Agent {
    async fn detections(&self, fresh: bool) -> Vec<Detection> {
        let mut g = self.engines.lock().await;
        if fresh || g.is_none() {
            let d = tendril_engines::detect_all().await;
            (self.ev)(AgentEvent::Engines {
                found: d.iter().map(|x| (x.engine.clone(), x.ready)).collect(),
            });
            *g = Some(d);
        }
        g.clone().unwrap_or_default()
    }

    async fn session(self: Arc<Self>, machine: u64) -> Result<End> {
        let conn = wire::connect(&self.opts.server, &self.psk).await?;
        let (mut reader, mut writer) = (conn.reader, conn.writer);
        let opts = self.opts.clone();
        let profile = tokio::task::spawn_blocking(move || {
            crate::machine::this_machine(opts.name.as_deref(), opts.max_memory)
        })
        .await?;
        let engines = self.detections(false).await;
        writer
            .send(&Msg::Hello {
                protocol: PROTOCOL,
                version: env!("CARGO_PKG_VERSION").into(),
                profile,
                tunnel_port: self.tunnel_port,
                machine,
                local: self.opts.local,
                engines: serde_json::to_string(&engines)?,
            })
            .await?;
        let lan_ip = match reader.recv().await? {
            Msg::Welcome {
                key,
                cluster,
                address,
            } => {
                (self.ev)(AgentEvent::Connected { name: key, cluster });
                address
            }
            Msg::Reject { reason } => return Ok(End::Rejected(reason)),
            other => bail!("unexpected reply {other:?}"),
        };
        let (out, mut out_rx) = mpsc::unbounded_channel::<Msg>();
        let writer_task = tokio::spawn(async move {
            while let Some(m) = out_rx.recv().await {
                if writer.send(&m).await.is_err() {
                    break;
                }
            }
        });
        let result = loop {
            let msg = match tokio::time::timeout(Duration::from_secs(20), reader.recv()).await {
                Ok(Ok(m)) => m,
                Ok(Err(e)) => break Err(e),
                Err(_) => break Err(anyhow::anyhow!("server stopped responding")),
            };
            match msg {
                Msg::Ping { t } => {
                    let _ = out.send(Msg::Pong { t });
                }
                Msg::Detect => {
                    let (me, out) = (self.clone(), out.clone());
                    tokio::spawn(async move {
                        let d = me.detections(true).await;
                        let _ = out.send(Msg::Engines {
                            engines: serde_json::to_string(&d).unwrap_or_default(),
                        });
                    });
                }
                Msg::Start {
                    deployment,
                    role,
                    peers,
                    namespace,
                } => {
                    let role: Role = match serde_json::from_str(&role) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::warn!("bad role from server: {e}");
                            continue;
                        }
                    };
                    self.start(
                        deployment,
                        role,
                        peers,
                        namespace,
                        lan_ip.clone(),
                        out.clone(),
                    );
                }
                Msg::Stop { deployment, role } => self.stop(&format!("{deployment}/{role}")),
                Msg::Bye { reason, leave } => break Ok(End::Bye { reason, leave }),
                _ => {}
            }
        };
        writer_task.abort();
        result
    }

    fn start(
        self: &Arc<Self>,
        deployment: String,
        role: Role,
        peers: Vec<Peer>,
        namespace: String,
        lan_ip: String,
        out: mpsc::UnboundedSender<Msg>,
    ) {
        let key = format!("{deployment}/{}", role.id);
        self.stop(&key);
        let pid = Arc::new(Mutex::new(None));
        let me = self.clone();
        let pid2 = pid.clone();
        let task = tokio::spawn(async move {
            let report = {
                let (out, ev) = (out.clone(), me.ev.clone());
                let (deployment, id, engine) = (
                    deployment.clone(),
                    role.id.clone(),
                    role.launch.engine.clone(),
                );
                move |state: RoleState| {
                    ev(AgentEvent::Role {
                        deployment: deployment.clone(),
                        role: id.clone(),
                        engine: engine.clone(),
                        state: state.clone(),
                    });
                    let _ = out.send(Msg::RoleUpdate {
                        deployment: deployment.clone(),
                        role: id.clone(),
                        state,
                    });
                }
            };
            let log = Arc::new(Mutex::new(VecDeque::new()));
            let ctx = RoleCtx {
                agent: me.clone(),
                deployment: deployment.clone(),
                role,
                peers,
                namespace,
                lan_ip,
                pid: pid2,
                log: log.clone(),
                report: Arc::new(report.clone()),
            };
            let key = format!("{}/{}", ctx.deployment, ctx.role.id);
            let r = ctx.run().await;
            me.targets.remove(&key);
            let tail = log.lock().unwrap().iter().cloned().collect::<Vec<_>>();
            match r {
                Ok(code) => report(RoleState::Exited { code, log: tail }),
                Err(e) => report(RoleState::Failed {
                    error: format!("{e:#}"),
                    log: tail,
                }),
            }
        });
        self.roles
            .lock()
            .unwrap()
            .insert(key, Running { task, pid });
    }

    fn stop(&self, key: &str) {
        let r = self.roles.lock().unwrap().remove(key);
        self.targets.remove(key);
        if let Some(r) = r {
            let pid = *r.pid.lock().unwrap();
            r.task.abort();
            if let Some(pid) = pid {
                kill_group(pid);
            }
        }
    }

    fn stop_all(&self) {
        let keys: Vec<String> = self.roles.lock().unwrap().keys().cloned().collect();
        for k in keys {
            self.stop(&k);
        }
    }
}

/// Stop an engine process and everything it started.
fn kill_group(pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::killpg(pid as i32, libc::SIGTERM);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(5));
            libc::killpg(pid as i32, libc::SIGKILL);
        });
    }
    #[cfg(not(unix))]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output();
    }
}

struct RoleCtx {
    agent: Arc<Agent>,
    deployment: String,
    role: Role,
    peers: Vec<Peer>,
    namespace: String,
    lan_ip: String,
    pid: Arc<Mutex<Option<u32>>>,
    log: Arc<Mutex<VecDeque<String>>>,
    report: Arc<dyn Fn(RoleState) + Send + Sync>,
}

/// Values the placeholders of a launch expand to on this machine.
#[derive(Default)]
pub struct Vars {
    pub ports: Vec<(String, u16)>,
    pub files: Vec<String>,
    /// Role → local end of its tunnel.
    pub peers: Vec<(String, String)>,
    pub peer_info: Vec<Peer>,
    pub dir: String,
    pub namespace: String,
    pub host: String,
    pub cache: String,
    pub lan_ip: String,
}

impl Vars {
    /// Expand `{…}` placeholders; unknown ones are an error (a typo would
    /// otherwise reach the engine as a literal).
    pub fn expand(&self, s: &str) -> Result<String> {
        let mut out = String::new();
        let mut rest = s;
        while let Some(i) = rest.find('{') {
            out.push_str(&rest[..i]);
            let Some(j) = rest[i..].find('}') else {
                out.push_str(&rest[i..]);
                return Ok(out);
            };
            let name = &rest[i + 1..i + j];
            out.push_str(
                &self
                    .value(name)
                    .with_context(|| format!("unknown placeholder {{{name}}}"))?,
            );
            rest = &rest[i + j + 1..];
        }
        out.push_str(rest);
        Ok(out)
    }

    fn value(&self, name: &str) -> Option<String> {
        let port = |n: &str| {
            self.ports
                .iter()
                .find(|(k, _)| k == n)
                .map(|(_, p)| p.to_string())
        };
        let peer = |r: &str| self.peer_info.iter().find(|p| p.role == r);
        let mut parts = name.splitn(3, ':');
        let (a, b, c) = (parts.next()?, parts.next(), parts.next());
        Some(match (a, b) {
            ("port", None) => port("")?,
            ("port", Some(n)) => port(n)?,
            ("file", Some(n)) => self.files.get(n.parse::<usize>().ok()?)?.clone(),
            ("peer", Some(r)) => self.peers.iter().find(|(k, _)| k == r)?.1.clone(),
            ("peer_host", Some(r)) => peer(r)?.host.clone(),
            ("peer_port", Some(r)) => {
                let want = c.unwrap_or("");
                peer(r)?
                    .ports
                    .iter()
                    .find(|(k, _)| k == want)?
                    .1
                    .to_string()
            }
            ("dir", None) => self.dir.clone(),
            ("namespace", None) => self.namespace.clone(),
            ("host", None) => self.host.clone(),
            ("cache", None) => self.cache.clone(),
            ("lan_ip", None) => self.lan_ip.clone(),
            _ => return None,
        })
    }
}

/// A relative path that stays inside its directory.
fn safe_rel(name: &str) -> Result<PathBuf> {
    let p = Path::new(name);
    if name.is_empty()
        || p.is_absolute()
        || p.components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        bail!("refusing file name {name:?}");
    }
    Ok(p.to_path_buf())
}

impl RoleCtx {
    /// The server sees its own machine as 127.0.0.1; other machines reach it
    /// at the address they use for the server.
    fn reachable(&self, p: &Peer) -> Peer {
        let mut p = p.clone();
        if !self.agent.opts.local && p.host == "127.0.0.1" {
            let server_host = self
                .agent
                .opts
                .server
                .rsplit_once(':')
                .map(|(h, _)| h.trim_matches(['[', ']']).to_string())
                .unwrap_or_default();
            if let Some((_, port)) = p.tunnel.rsplit_once(':') {
                p.tunnel = format!("{server_host}:{port}");
            }
            p.host = server_host;
        }
        p
    }

    fn say(&self, s: RoleState) {
        (self.report)(s)
    }

    /// Run the role until its process exits (Ok(exit code)) or fails.
    async fn run(&self) -> Result<Option<i32>> {
        let launch = &self.role.launch;
        let engine = tendril_engines::get(&launch.engine)
            .with_context(|| format!("this Tendril doesn't know the engine {}", launch.engine))?;
        let located = engine
            .locate()
            .await
            .with_context(|| format!("{} is not installed on this machine", engine.info().name))?;
        let cmd = located
            .program(&launch.program)
            .with_context(|| {
                format!(
                    "{} has no {} on this machine",
                    engine.info().name,
                    launch.program
                )
            })?
            .clone();
        let cache = &self.agent.opts.cache;
        let engine_cache = cache
            .join("engines")
            .join(tendril_engines::util::safe_name(&launch.engine));
        // Files first: models can be large, and the process needs them.
        let mut files = Vec::new();
        for f in &launch.files {
            let dest = match &f.engine_dir {
                Some(d) => located
                    .dir
                    .as_ref()
                    .context("the engine has no install directory for its model files")?
                    .join(safe_rel(d)?)
                    .join(safe_rel(
                        Path::new(&f.name)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(""),
                    )?),
                None => cache.join("models").join(safe_rel(&f.name)?),
            };
            let (url, size, report) = (f.url.clone(), f.size, self.report.clone());
            let label = f.name.clone();
            let d2 = dest.clone();
            tokio::task::spawn_blocking(move || {
                let mut last = Instant::now() - Duration::from_secs(1);
                fetch::download_file(&url, size, &d2, &mut |done, total| {
                    if last.elapsed() > Duration::from_millis(500) || done == total {
                        last = Instant::now();
                        report(RoleState::Downloading {
                            file: label.clone(),
                            done,
                            total,
                        });
                    }
                })
            })
            .await??;
            files.push(dest.display().to_string());
        }
        self.say(RoleState::Starting);
        let mut vars = Vars {
            files,
            dir: located
                .dir
                .as_ref()
                .map(|d| d.display().to_string())
                .unwrap_or_default(),
            namespace: self.namespace.clone(),
            host: if launch.lan {
                "0.0.0.0".into()
            } else {
                "127.0.0.1".into()
            },
            cache: engine_cache.display().to_string(),
            lan_ip: self.lan_ip.clone(),
            peer_info: self.peers.iter().map(|p| self.reachable(p)).collect(),
            ..Default::default()
        };
        std::fs::create_dir_all(&engine_cache).ok();
        vars.ports
            .push((String::new(), tendril_engines::detect::free_port()?));
        for n in &launch.ports {
            vars.ports
                .push((n.clone(), tendril_engines::detect::free_port()?));
        }
        // Local ends of encrypted tunnels to the roles this one talks to.
        let mut forwarders = Vec::new();
        for r in &launch.peers {
            let p = self
                .peers
                .iter()
                .find(|p| &p.role == r)
                .map(|p| self.reachable(p))
                .with_context(|| format!("{r} has not started"))?;
            let p = &p;
            let main = p
                .ports
                .iter()
                .find(|(k, _)| k.is_empty())
                .map(|(_, v)| *v)
                .unwrap_or(0);
            let route = if p.tunnel == format!("{}:{}", self.lan_ip, self.agent.tunnel_port)
                || (self.agent.opts.local
                    && p.tunnel == format!("127.0.0.1:{}", self.agent.tunnel_port))
            {
                Route::Local(main)
            } else {
                Route::Remote {
                    tunnel: p.tunnel.clone(),
                    target: format!("{}/{r}", self.deployment),
                }
            };
            let f = Forwarder::start(route, self.agent.psk).await?;
            vars.peers
                .push((r.clone(), format!("127.0.0.1:{}", f.port)));
            forwarders.push(f);
        }
        let args: Vec<String> = launch
            .args
            .iter()
            .map(|a| vars.expand(a))
            .collect::<Result<_>>()?;
        let env: Vec<(String, String)> = launch
            .env
            .iter()
            .map(|(k, v)| Ok((k.clone(), vars.expand(v)?)))
            .collect::<Result<_>>()?;
        let mut c = tokio::process::Command::new(&cmd[0]);
        c.args(&cmd[1..])
            .args(&args)
            .envs(env)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(d) = &launch.cwd {
            c.current_dir(vars.expand(d)?);
        }
        #[cfg(unix)]
        c.process_group(0);
        tracing::info!("{}: {} {}", self.role.id, cmd.join(" "), args.join(" "));
        let mut child = c
            .spawn()
            .with_context(|| format!("cannot start {}", cmd[0]))?;
        *self.pid.lock().unwrap() = child.id();
        for s in [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn tokio::io::AsyncRead + Unpin + Send>),
            child.stderr.take().map(|s| Box::new(s) as _),
        ]
        .into_iter()
        .flatten()
        {
            let log = self.log.clone();
            let id = self.role.id.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(s).lines();
                while let Ok(Some(l)) = lines.next_line().await {
                    tracing::debug!("[{id}] {l}");
                    let mut g = log.lock().unwrap();
                    g.push_back(l);
                    while g.len() > 60 {
                        g.pop_front();
                    }
                }
            });
        }
        let main = vars.ports[0].1;
        let key = format!("{}/{}", self.deployment, self.role.id);
        self.agent.targets.insert(&key, main);
        self.say(RoleState::Started {
            ports: vars.ports.clone(),
        });
        if let Some(code) = self.wait_ready(launch, main, &mut child).await? {
            return Ok(code);
        }
        if self.role.serves_api {
            let mut expanded = launch.clone();
            expanded.args = args;
            let report = self.report.clone();
            let progress: tendril_engines::Progress =
                Arc::new(move |what: &str, done: u64, total: u64| {
                    report(RoleState::Preparing {
                        what: what.to_string(),
                        done,
                        total,
                    })
                });
            let base = format!("http://127.0.0.1:{main}");
            tokio::select! {
                r = engine.prepare(&expanded, &base, progress) => r?,
                s = child.wait() => bail!("the engine exited while preparing ({})", s?),
            }
        }
        self.say(RoleState::Ready);
        let status = child.wait().await?;
        drop(forwarders);
        Ok(status.code())
    }

    /// Wait until the process is ready. Some(code) if it exited first.
    async fn wait_ready(
        &self,
        launch: &Launch,
        port: u16,
        child: &mut tokio::process::Child,
    ) -> Result<Option<Option<i32>>> {
        let t0 = Instant::now();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()?;
        loop {
            if let Some(st) = child.try_wait()? {
                let tail = self
                    .log
                    .lock()
                    .unwrap()
                    .iter()
                    .rev()
                    .take(5)
                    .rev()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" | ");
                bail!(
                    "the engine exited before it was ready ({st}){}",
                    if tail.is_empty() {
                        String::new()
                    } else {
                        format!(": {tail}")
                    }
                );
            }
            let ok = match &launch.ready {
                Ready::Http { path } => client
                    .get(format!("http://127.0.0.1:{port}{path}"))
                    .send()
                    .await
                    .is_ok_and(|r| r.status().is_success()),
                Ready::Port => tokio::net::TcpStream::connect(("127.0.0.1", port))
                    .await
                    .is_ok(),
                Ready::Started => t0.elapsed() > Duration::from_secs(3),
            };
            if ok {
                return Ok(None);
            }
            if t0.elapsed() > Duration::from_secs(launch.ready_timeout.max(5)) {
                bail!("the engine was not ready after {} s", launch.ready_timeout);
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders() {
        let v = Vars {
            ports: vec![("".into(), 5000), ("master".into(), 5001)],
            files: vec!["/c/m.gguf".into()],
            peers: vec![("rpc-1".into(), "127.0.0.1:6000".into())],
            peer_info: vec![Peer {
                role: "server".into(),
                tunnel: "10.0.0.2:7001".into(),
                host: "10.0.0.2".into(),
                ports: vec![("master".into(), 29500)],
            }],
            namespace: "ns".into(),
            host: "127.0.0.1".into(),
            ..Default::default()
        };
        assert_eq!(
            v.expand("-m {file:0} --port {port}").unwrap(),
            "-m /c/m.gguf --port 5000"
        );
        assert_eq!(
            v.expand("{peer:rpc-1},{port:master}").unwrap(),
            "127.0.0.1:6000,5001"
        );
        assert_eq!(
            v.expand("{peer_host:server}:{peer_port:server:master}")
                .unwrap(),
            "10.0.0.2:29500"
        );
        assert!(v.expand("{nope}").is_err());
        assert!(v.expand("{file:3}").is_err());
        assert_eq!(v.expand("{\"json\": 1").unwrap(), "{\"json\": 1");
        assert!(safe_rel("../x").is_err());
        assert!(safe_rel("/etc/passwd").is_err());
        assert!(safe_rel("org--m/sub/m.gguf").is_ok());
    }
}
