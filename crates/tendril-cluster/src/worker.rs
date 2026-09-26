//! Runs one pipeline stage on a dedicated thread.
//!
//! Work arrives in order on a queue (forwards, releases, upstream errors);
//! results go to the next hop. The last stage samples tokens itself so only a
//! token id — not a vocabulary-sized logits vector — crosses the network.

use crate::proto::{Msg, Payload, WireTensor};
use std::collections::HashMap;
use std::sync::mpsc;
use tendril_engine::model::{Stage, StageInput, StageOutput};
use tendril_engine::sampler::Sampler;
use tokio::sync::mpsc::UnboundedSender;

pub enum Work {
    Msg(Msg),
    Stop,
}

pub struct StageWorker {
    pub tx: mpsc::Sender<Work>,
    pub epoch: u64,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl StageWorker {
    pub fn spawn(stage: Stage, index: u32, epoch: u64, out: UnboundedSender<Msg>) -> StageWorker {
        let (tx, rx) = mpsc::channel::<Work>();
        let handle = std::thread::Builder::new()
            .name(format!("tendril-stage-{index}"))
            .spawn(move || run(stage, index, epoch, rx, out))
            .expect("spawn stage thread");
        StageWorker {
            tx,
            epoch,
            handle: Some(handle),
        }
    }

    pub fn submit(&self, m: Msg) -> bool {
        self.tx.send(Work::Msg(m)).is_ok()
    }
}

impl Drop for StageWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(Work::Stop);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run(
    mut stage: Stage,
    index: u32,
    epoch: u64,
    rx: mpsc::Receiver<Work>,
    out: UnboundedSender<Msg>,
) {
    let mut samplers: HashMap<u64, Sampler> = HashMap::new();
    while let Ok(w) = rx.recv() {
        let msg = match w {
            Work::Msg(m) => m,
            Work::Stop => break,
        };
        match msg {
            Msg::Forward {
                epoch: e,
                seq,
                pos,
                payload,
                want_logits,
                sample,
            } => {
                if e != epoch {
                    continue; // stale work from a previous plan
                }
                if stage.spec.head {
                    if let Some(s) = &sample {
                        samplers.insert(seq, Sampler::new(s.params.clone(), &s.history));
                    }
                }
                let result = (|| -> anyhow::Result<Option<Msg>> {
                    let n_in;
                    let input = match payload {
                        Payload::Tokens(t) => {
                            n_in = t.len();
                            StageInput::Tokens(t)
                        }
                        Payload::Hidden(h) => {
                            let t = h.to_tensor(&stage.device)?;
                            n_in = t.dim(1)?;
                            StageInput::Hidden(t)
                        }
                    };
                    match stage.forward(seq, pos as usize, input, want_logits)? {
                        StageOutput::Hidden(h) => Ok(Some(Msg::Forward {
                            epoch,
                            seq,
                            pos,
                            payload: Payload::Hidden(WireTensor::from_tensor(&h)?),
                            want_logits,
                            sample,
                        })),
                        StageOutput::Logits(mut l) => {
                            let s = samplers
                                .get_mut(&seq)
                                .ok_or_else(|| anyhow::anyhow!("no sampler for sequence {seq}"))?;
                            let token = s.sample(&mut l);
                            Ok(Some(Msg::Token {
                                epoch,
                                seq,
                                pos: pos + n_in as u32 - 1,
                                token,
                            }))
                        }
                        StageOutput::Nothing => Ok(None),
                    }
                })();
                match result {
                    Ok(Some(m)) => {
                        let _ = out.send(m);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        stage.release(seq);
                        samplers.remove(&seq);
                        let _ = out.send(Msg::StageError {
                            epoch,
                            seq,
                            stage: index,
                            error: format!("{e:#}"),
                        });
                    }
                }
            }
            Msg::Release { epoch: e, seq } => {
                stage.release(seq);
                samplers.remove(&seq);
                let _ = out.send(Msg::Release { epoch: e, seq });
            }
            Msg::StageError {
                epoch: e,
                seq,
                stage: s,
                error,
            } => {
                stage.release(seq);
                samplers.remove(&seq);
                let _ = out.send(Msg::StageError {
                    epoch: e,
                    seq,
                    stage: s,
                    error,
                });
            }
            _ => {}
        }
    }
}
