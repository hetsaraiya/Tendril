//! Runs one pipeline stage on a dedicated thread.
//!
//! Work arrives in order on a queue (forwards, releases, upstream errors).
//! Whatever forwards are waiting when the stage becomes free — decode steps of
//! different conversations and prefill chunks — run together as one batch, so
//! every weight is read once per batch instead of once per request
//! (continuous batching). The last stage samples tokens itself so only a token
//! id crosses the network.

use crate::proto::{Draft, KvOpKind, Msg, Payload, SampleSetup, StageTime, WireTensor};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::time::Instant;
use tendril_engine::model::{BatchItem, Stage, StageInput, StageOutput};
use tendril_engine::sampler::Sampler;
use tokio::sync::mpsc::UnboundedSender;

/// Most tokens processed in one stage pass (a prefill chunk plus decodes).
pub const MAX_BATCH_TOKENS: usize = 640;
/// Most sequences in one stage pass.
pub const MAX_BATCH_SEQS: usize = 64;

pub enum Work {
    /// A message and when it arrived.
    Msg(Msg, Instant),
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
        let kv_dir = crate::shard::default_cache()
            .join("kv")
            .join(format!("{}-{epoch}-{index}", std::process::id()));
        let handle = std::thread::Builder::new()
            .name(format!("tendril-stage-{index}"))
            .spawn(move || {
                run(stage, index, epoch, rx, out, &kv_dir);
                let _ = std::fs::remove_dir_all(&kv_dir);
            })
            .expect("spawn stage thread");
        StageWorker {
            tx,
            epoch,
            handle: Some(handle),
        }
    }

    pub fn submit(&self, m: Msg) -> bool {
        self.tx.send(Work::Msg(m, Instant::now())).is_ok()
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

struct Pending {
    seq: u64,
    pos: u32,
    payload: Payload,
    want_logits: bool,
    sample: Option<SampleSetup>,
    trace: Vec<StageTime>,
    draft: Option<Draft>,
    arrived: Instant,
}

fn tokens_of(p: &Payload) -> usize {
    match p {
        Payload::Tokens(t) => t.len(),
        Payload::Hidden(h) => h.shape.get(1).copied().unwrap_or(1),
    }
}

fn run(
    mut stage: Stage,
    index: u32,
    epoch: u64,
    rx: mpsc::Receiver<Work>,
    out: UnboundedSender<Msg>,
    kv_dir: &std::path::Path,
) {
    let mut samplers: HashMap<u64, Sampler> = HashMap::new();
    let mut queue: VecDeque<(Msg, Instant)> = VecDeque::new();
    let mut stopping = false;
    loop {
        if queue.is_empty() {
            if stopping {
                break;
            }
            match rx.recv() {
                Ok(Work::Msg(m, t)) => queue.push_back((m, t)),
                Ok(Work::Stop) | Err(_) => break,
            }
        }
        // Everything that arrived while we were busy joins this round.
        while let Ok(w) = rx.try_recv() {
            match w {
                Work::Msg(m, t) => queue.push_back((m, t)),
                Work::Stop => stopping = true,
            }
        }
        let Some((front, _)) = queue.front() else {
            continue;
        };
        if !matches!(front, Msg::Forward { .. }) {
            let (m, _) = queue.pop_front().unwrap();
            control(&mut stage, &mut samplers, m, &out, kv_dir);
            continue;
        }
        // Form a batch: consecutive forwards, one item per sequence, within budget.
        let mut batch: Vec<Pending> = Vec::new();
        let mut tokens = 0;
        let mut i = 0;
        while i < queue.len() && batch.len() < MAX_BATCH_SEQS {
            let (m, _) = &queue[i];
            let Msg::Forward {
                epoch: e,
                seq,
                payload,
                ..
            } = m
            else {
                break;
            };
            if *e != epoch {
                queue.remove(i); // stale work from a previous plan
                continue;
            }
            let n = tokens_of(payload);
            if batch.iter().any(|b| b.seq == *seq)
                || (!batch.is_empty() && tokens + n > MAX_BATCH_TOKENS)
            {
                i += 1;
                continue;
            }
            let (m, arrived) = queue.remove(i).unwrap();
            if let Msg::Forward {
                seq,
                pos,
                payload,
                want_logits,
                sample,
                trace,
                draft,
                ..
            } = m
            {
                tokens += n;
                batch.push(Pending {
                    draft,
                    seq,
                    pos,
                    payload,
                    want_logits,
                    sample,
                    trace,
                    arrived,
                });
            }
        }
        if !batch.is_empty() {
            execute(&mut stage, &mut samplers, index, epoch, batch, &out);
        }
    }
}

fn control(
    stage: &mut Stage,
    samplers: &mut HashMap<u64, Sampler>,
    m: Msg,
    out: &UnboundedSender<Msg>,
    kv_dir: &std::path::Path,
) {
    match m {
        Msg::KvOp {
            epoch,
            seq,
            op,
            ok,
            error,
        } => {
            let path = kv_dir.join(format!("{seq}.kv"));
            let r: anyhow::Result<()> = if !ok {
                // An earlier stage failed: keep this stage consistent by dropping it.
                stage.release(seq);
                Ok(())
            } else {
                match op {
                    KvOpKind::Truncate { len } => stage.truncate(seq, len as usize),
                    KvOpKind::Spill => stage.spill(seq, &path).map(|_| ()),
                    KvOpKind::Restore => stage
                        .restore(seq, &path)
                        .and_then(|_| std::fs::remove_file(&path).map_err(Into::into)),
                    KvOpKind::Drop => {
                        stage.release(seq);
                        let _ = std::fs::remove_file(&path);
                        Ok(())
                    }
                }
            };
            // Truncation keeps the conversation (speculative rollback, prefix
            // reuse); moving it out of memory ends this stage's sampling state.
            if !matches!(op, KvOpKind::Truncate { .. }) {
                samplers.remove(&seq);
            }
            let (ok, error) = match r {
                Ok(()) => (ok, error),
                Err(e) => (false, Some(format!("stage: {e:#}"))),
            };
            let _ = out.send(Msg::KvOp {
                epoch,
                seq,
                op,
                ok,
                error,
            });
        }
        Msg::Release { epoch, seq } => {
            stage.release(seq);
            samplers.remove(&seq);
            let _ = out.send(Msg::Release { epoch, seq });
        }
        Msg::StageError {
            epoch,
            seq,
            stage: s,
            error,
        } => {
            stage.release(seq);
            samplers.remove(&seq);
            let _ = out.send(Msg::StageError {
                epoch,
                seq,
                stage: s,
                error,
            });
        }
        _ => {}
    }
}

fn execute(
    stage: &mut Stage,
    samplers: &mut HashMap<u64, Sampler>,
    index: u32,
    epoch: u64,
    batch: Vec<Pending>,
    out: &UnboundedSender<Msg>,
) {
    let started = Instant::now();
    let size = batch.len() as u16;
    // Decode inputs; a malformed payload fails only its own sequence.
    let mut items = Vec::with_capacity(batch.len());
    let mut meta = Vec::with_capacity(batch.len());
    for p in batch {
        if stage.spec.head {
            if let Some(s) = &p.sample {
                samplers.insert(p.seq, Sampler::new(s.params.clone(), &s.history));
            }
        }
        let input = match p.payload {
            Payload::Tokens(t) => Ok(StageInput::Tokens(t)),
            Payload::Hidden(h) => h.to_tensor(&stage.device).map(StageInput::Hidden),
        };
        match input {
            Ok(input) => {
                let n = match &input {
                    StageInput::Tokens(t) => t.len(),
                    StageInput::Hidden(h) => h.dim(1).unwrap_or(1),
                };
                items.push(BatchItem {
                    seq: p.seq,
                    pos: p.pos as usize,
                    input,
                    want_logits: p.want_logits,
                    all_logits: stage.spec.head && p.draft.is_some(),
                });
                meta.push((
                    p.seq,
                    p.pos,
                    n,
                    p.want_logits,
                    p.sample,
                    p.trace,
                    p.draft,
                    p.arrived,
                ));
            }
            Err(e) => fail(stage, samplers, index, epoch, p.seq, format!("{e:#}"), out),
        }
    }
    let results = stage.forward_batch(items);
    let compute_us = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
    for (r, (seq, pos, n, want_logits, sample, mut trace, draft, arrived)) in
        results.into_iter().zip(meta)
    {
        trace.push(StageTime {
            stage: index,
            queue_us: started
                .duration_since(arrived)
                .as_micros()
                .min(u32::MAX as u128) as u32,
            compute_us,
            batch: size,
        });
        let msg = match r {
            Ok(StageOutput::Hidden(h)) => match WireTensor::from_tensor(&h) {
                Ok(t) => Some(Msg::Forward {
                    epoch,
                    seq,
                    pos,
                    payload: Payload::Hidden(t),
                    want_logits,
                    sample,
                    trace,
                    draft,
                }),
                Err(e) => {
                    fail(stage, samplers, index, epoch, seq, format!("{e:#}"), out);
                    None
                }
            },
            Ok(StageOutput::Logits(mut l)) => match samplers.get_mut(&seq) {
                Some(s) => Some(Msg::Token {
                    epoch,
                    seq,
                    pos: pos + n as u32 - 1,
                    token: s.sample(&mut l),
                    trace,
                }),
                None => {
                    fail(
                        stage,
                        samplers,
                        index,
                        epoch,
                        seq,
                        format!("no sampler for sequence {seq}"),
                        out,
                    );
                    None
                }
            },
            Ok(StageOutput::AllLogits(mut rows)) => match (samplers.get_mut(&seq), draft) {
                (Some(s), Some(d)) if rows.len() == d.tokens.len() + 1 => {
                    let tokens = s.verify(&mut rows, &d.tokens, &d.q);
                    Some(Msg::Tokens {
                        epoch,
                        seq,
                        pos,
                        tokens,
                        trace,
                    })
                }
                _ => {
                    fail(
                        stage,
                        samplers,
                        index,
                        epoch,
                        seq,
                        format!("sequence {seq}: malformed speculative draft"),
                        out,
                    );
                    None
                }
            },
            Ok(StageOutput::Nothing) => None,
            Err(e) => {
                fail(stage, samplers, index, epoch, seq, format!("{e:#}"), out);
                None
            }
        };
        if let Some(m) = msg {
            let _ = out.send(m);
        }
    }
}

fn fail(
    stage: &mut Stage,
    samplers: &mut HashMap<u64, Sampler>,
    index: u32,
    epoch: u64,
    seq: u64,
    error: String,
    out: &UnboundedSender<Msg>,
) {
    stage.release(seq);
    samplers.remove(&seq);
    let _ = out.send(Msg::StageError {
        epoch,
        seq,
        stage: index,
        error,
    });
}
