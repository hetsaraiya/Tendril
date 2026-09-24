//! `tendril join`: contribute this machine to a cluster.

use crate::proto::{Msg, NextHop, PROTOCOL};
use crate::shard::{is_complete, shard_path, ShardWriter};
use crate::worker::StageWorker;
use crate::{token, wire};
use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tendril_engine::config::ModelConfig;
use tendril_engine::device::{activation_dtype, device_from_name, device_label};
use tendril_engine::linear::WeightFormat;
use tendril_engine::model::{LoadOptions, Stage, StageSpec};
use tendril_engine::weights::WeightStore;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct AgentOptions {
    pub coordinator: String,
    pub token: String,
    pub name: Option<String>,
    pub device: String,
    pub data_port: u16,
    pub cache: PathBuf,
    /// Stop instead of reconnecting when the coordinator goes away.
    pub once: bool,
    /// Cap on memory Tendril may use here.
    pub max_memory: Option<tendril_core::Bytes>,
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
    Receiving {
        spec: StageSpec,
        done: u64,
        total: u64,
    },
    Loading {
        spec: StageSpec,
        cached: bool,
    },
    Ready {
        spec: StageSpec,
        weight_bytes: u64,
        load_ms: u64,
        device: String,
    },
    Unloaded,
    Failed {
        error: String,
    },
}

pub type EventFn = Arc<dyn Fn(AgentEvent) + Send + Sync>;

type Active = Arc<Mutex<Option<StageWorker>>>;

pub async fn run(opts: AgentOptions, ev: EventFn) -> Result<()> {
    let psk = token::psk(&opts.token);
    let listener = TcpListener::bind(("0.0.0.0", opts.data_port))
        .await
        .with_context(|| format!("cannot listen on data port {}", opts.data_port))?;
    let data_port = listener.local_addr()?.port();
    let active: Active = Arc::new(Mutex::new(None));
    tokio::spawn(data_loop(listener, psk, active.clone()));
    let mut backoff = Duration::from_secs(1);
    loop {
        ev(AgentEvent::Connecting {
            addr: opts.coordinator.clone(),
        });
        let started = Instant::now();
        let r = session(&opts, &psk, data_port, active.clone(), &ev).await;
        drop(active.lock().unwrap().take());
        match r {
            Ok(Some(reason)) => {
                ev(AgentEvent::Rejected {
                    reason: reason.clone(),
                });
                if opts.once {
                    bail!(reason);
                }
            }
            Ok(None) if opts.once => return Ok(()),
            Ok(None) => {}
            Err(e) => {
                if started.elapsed() > Duration::from_secs(30) {
                    backoff = Duration::from_secs(1);
                }
                let msg = format!("{e:#}");
                if msg.contains("wrong cluster token") || msg.contains("authentication") {
                    ev(AgentEvent::Rejected {
                        reason: "the coordinator rejected our token — check `--token`".into(),
                    });
                    bail!("wrong cluster token");
                }
                if opts.once {
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

/// One control session. Ok(Some(reason)) = rejected, Ok(None) = coordinator said bye.
async fn session(
    opts: &AgentOptions,
    psk: &[u8; 32],
    data_port: u16,
    active: Active,
    ev: &EventFn,
) -> Result<Option<String>> {
    let conn = wire::connect(&opts.coordinator, psk).await?;
    let (mut reader, mut writer) = (conn.reader, conn.writer);
    let mut profile = tendril_core::hardware::detect_local(true);
    if let Some(n) = &opts.name {
        profile.name = n.clone();
    }
    // Report the backend this build can really use.
    let dev = device_from_name(&opts.device)?;
    if matches!(dev, candle_core::Device::Cpu) && profile.backend != tendril_core::Backend::Cpu {
        profile.backend = tendril_core::Backend::Cpu;
        profile.recompute_usable();
    }
    crate::coordinator::apply_memory_cap(&mut profile, opts.max_memory);
    writer
        .send(&Msg::Hello {
            protocol: PROTOCOL,
            version: env!("CARGO_PKG_VERSION").into(),
            profile,
            data_port,
            backends: tendril_engine::device::compiled_backends()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })
        .await?;
    match reader.recv().await? {
        Msg::Welcome { name, cluster, .. } => ev(AgentEvent::Connected { name, cluster }),
        Msg::Reject { reason } => return Ok(Some(reason)),
        other => bail!("unexpected reply {other:?}"),
    }
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Msg>();
    let writer_task = tokio::spawn(async move {
        while let Some(m) = out_rx.recv().await {
            if writer.send(&m).await.is_err() {
                break;
            }
        }
    });
    let mut weights_tx: Option<mpsc::UnboundedSender<Msg>> = None;
    let result = loop {
        let msg = match tokio::time::timeout(Duration::from_secs(20), reader.recv()).await {
            Ok(Ok(m)) => m,
            Ok(Err(e)) => break Err(e),
            Err(_) => break Err(anyhow::anyhow!("coordinator stopped responding")),
        };
        match msg {
            Msg::Ping { t } => {
                let _ = out_tx.send(Msg::Pong { t });
            }
            Msg::Probe { data } => {
                let _ = out_tx.send(Msg::ProbeAck {
                    len: data.len() as u64,
                });
            }
            Msg::LoadStage {
                epoch,
                index,
                model_key,
                config_json,
                spec,
                format,
                device,
                tensors,
                next,
            } => {
                drop(active.lock().unwrap().take());
                let (wtx, wrx) = mpsc::unbounded_channel();
                weights_tx = Some(wtx);
                let job = LoadJob {
                    epoch,
                    index,
                    model_key,
                    config_json,
                    spec,
                    format,
                    device: if opts.device == "auto" {
                        device
                    } else {
                        opts.device.clone()
                    },
                    tensors,
                    next,
                };
                tokio::spawn(load_stage(
                    job,
                    psk.to_owned(),
                    opts.cache.clone(),
                    wrx,
                    out_tx.clone(),
                    active.clone(),
                    ev.clone(),
                ));
            }
            m @ (Msg::WeightData { .. } | Msg::WeightsDone { .. }) => {
                if let Some(w) = &weights_tx {
                    let _ = w.send(m);
                }
            }
            Msg::Unload { epoch } => {
                let mut a = active.lock().unwrap();
                if a.as_ref().is_some_and(|w| w.epoch == epoch) {
                    a.take();
                    ev(AgentEvent::Unloaded);
                }
            }
            Msg::Bye { .. } => break Ok(None),
            _ => {}
        }
    };
    writer_task.abort();
    result
}

struct LoadJob {
    epoch: u64,
    index: u32,
    model_key: String,
    config_json: String,
    spec: StageSpec,
    format: String,
    device: String,
    tensors: Vec<crate::proto::TensorEntry>,
    next: NextHop,
}

async fn load_stage(
    job: LoadJob,
    psk: [u8; 32],
    cache: PathBuf,
    mut wrx: mpsc::UnboundedReceiver<Msg>,
    out: mpsc::UnboundedSender<Msg>,
    active: Active,
    ev: EventFn,
) {
    let epoch = job.epoch;
    let spec = job.spec;
    let r: Result<()> = async {
        let t0 = Instant::now();
        let path = shard_path(&cache, &job.model_key, &job.tensors);
        let cached = is_complete(&path);
        if !cached {
            let names = job.tensors.iter().map(|t| t.name.clone()).collect();
            out.send(Msg::NeedWeights { epoch, names })?;
            let mut w = ShardWriter::create(&path, &job.tensors)?;
            let total = w.total;
            let mut last = Instant::now();
            loop {
                match wrx.recv().await {
                    Some(Msg::WeightData {
                        epoch: e,
                        name,
                        offset,
                        data,
                    }) if e == epoch => {
                        w.write(&name, offset, &data)?;
                        if last.elapsed() > Duration::from_millis(250) {
                            last = Instant::now();
                            let _ = out.send(Msg::LoadProgress {
                                epoch,
                                phase: "receiving".into(),
                                done: w.written(),
                                total,
                            });
                            ev(AgentEvent::Receiving {
                                spec,
                                done: w.written(),
                                total,
                            });
                        }
                    }
                    Some(Msg::WeightsDone { epoch: e }) if e == epoch => break,
                    Some(_) => {}
                    None => bail!("coordinator disconnected while sending weights"),
                }
            }
            ev(AgentEvent::Receiving {
                spec,
                done: w.written(),
                total,
            });
            w.finish()?;
        } else {
            out.send(Msg::NeedWeights {
                epoch,
                names: vec![],
            })?;
        }
        ev(AgentEvent::Loading { spec, cached });
        let _ = out.send(Msg::LoadProgress {
            epoch,
            phase: "loading".into(),
            done: 0,
            total: 0,
        });
        let cfg_json = job.config_json.clone();
        let fmt = job.format.clone();
        let device_name = job.device.clone();
        let stage = tokio::task::spawn_blocking(move || -> Result<Stage> {
            let raw: serde_json::Value = serde_json::from_str(&cfg_json)?;
            let cfg = Arc::new(ModelConfig::from_json(&raw)?);
            let ws = WeightStore::open(&[path])?;
            let device = device_from_name(&device_name)?;
            let dtype = activation_dtype(&device, &cfg.torch_dtype);
            let format = WeightFormat::parse(&fmt).unwrap_or(WeightFormat::Native);
            Stage::load(
                cfg,
                &ws,
                spec,
                &LoadOptions {
                    format,
                    device,
                    dtype,
                },
            )
        })
        .await??;
        let weight_bytes = stage.weight_bytes as u64;
        let device = device_label(&stage.device).to_string();
        // Connect to the next hop before accepting work.
        let addr = match &job.next {
            NextHop::Stage(a) | NextHop::Coordinator(a) => a.clone(),
        };
        let next = connect_retry(&addr, &psk, Duration::from_secs(60)).await?;
        let (tx, rx) = mpsc::unbounded_channel::<Msg>();
        tokio::spawn(pump(
            next,
            epoch,
            matches!(job.next, NextHop::Coordinator(_)),
            rx,
        ));
        let worker = StageWorker::spawn(stage, job.index, epoch, tx);
        *active.lock().unwrap() = Some(worker);
        let load_ms = t0.elapsed().as_millis() as u64;
        out.send(Msg::StageReady {
            epoch,
            weight_bytes,
            load_ms,
            device: device.clone(),
        })?;
        ev(AgentEvent::Ready {
            spec,
            weight_bytes,
            load_ms,
            device,
        });
        Ok(())
    }
    .await;
    if let Err(e) = r {
        let error = format!("{e:#}");
        ev(AgentEvent::Failed {
            error: error.clone(),
        });
        let _ = out.send(Msg::LoadFailed { epoch, error });
    }
}

pub(crate) async fn connect_retry(
    addr: &str,
    psk: &[u8; 32],
    within: Duration,
) -> Result<wire::Conn> {
    let start = Instant::now();
    loop {
        match wire::connect(addr, psk).await {
            Ok(c) => return Ok(c),
            Err(e) if start.elapsed() < within => {
                tracing::debug!("waiting for {addr}: {e:#}");
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
            Err(e) => return Err(e).with_context(|| format!("cannot reach next stage at {addr}")),
        }
    }
}

/// Forward a stage's output to the next hop.
pub(crate) async fn pump(
    mut conn: wire::Conn,
    epoch: u64,
    results: bool,
    mut rx: mpsc::UnboundedReceiver<Msg>,
) {
    if conn
        .writer
        .send(&Msg::DataHello { epoch, results })
        .await
        .is_err()
    {
        return;
    }
    while let Some(m) = rx.recv().await {
        if let Err(e) = conn.writer.send(&m).await {
            tracing::warn!("next hop connection lost: {e:#}");
            break;
        }
    }
}

/// Accept data connections and feed the active stage.
async fn data_loop(listener: TcpListener, psk: [u8; 32], active: Active) {
    loop {
        let Ok((s, _)) = listener.accept().await else {
            continue;
        };
        let active = active.clone();
        tokio::spawn(async move {
            let Ok(mut c) = wire::accept(s, &psk).await else {
                return;
            };
            let epoch = match c.reader.recv().await {
                Ok(Msg::DataHello { epoch, .. }) => epoch,
                _ => return,
            };
            while let Ok(m) = c.reader.recv().await {
                let guard = active.lock().unwrap();
                match guard.as_ref() {
                    Some(w) if w.epoch == epoch => {
                        w.submit(m);
                    }
                    _ => {}
                }
            }
        });
    }
}
