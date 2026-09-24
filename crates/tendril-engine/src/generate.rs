//! Generation on a single machine: all stages in-process.
//! Also home to pieces shared with the distributed coordinator.

use crate::config::ModelConfig;
use crate::device::activation_dtype;
use crate::linear::WeightFormat;
use crate::model::{LoadOptions, Stage, StageInput, StageOutput, StageSpec};
use crate::sampler::{Sampler, SamplingParams};
use crate::tokenizer::{Detokenizer, Tok};
use crate::weights::WeightStore;
use anyhow::{bail, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}

impl FinishReason {
    pub fn openai(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "stop",
            FinishReason::Error => "error",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Timing {
    pub prompt_tokens: usize,
    /// Prompt tokens served from the prefix cache (not recomputed).
    #[serde(default)]
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub queue_ms: f64,
    pub ttft_ms: f64,
    pub total_ms: f64,
}

impl Timing {
    pub fn decode_tps(&self) -> f64 {
        let decode_ms = self.total_ms - self.ttft_ms;
        if self.completion_tokens > 1 && decode_ms > 0.0 {
            (self.completion_tokens - 1) as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        }
    }
    pub fn prefill_tps(&self) -> f64 {
        if self.ttft_ms > 0.0 {
            self.prompt_tokens as f64 / (self.ttft_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// Holds back text that might be the start of a stop string.
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
    max_len: usize,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let max_len = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        StopMatcher {
            stops,
            pending: String::new(),
            max_len,
        }
    }

    /// Returns (text safe to emit, stopped).
    pub fn push(&mut self, delta: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (delta.to_string(), false);
        }
        self.pending.push_str(delta);
        if let Some((idx, _)) = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()).map(|i| (i, s)))
            .min_by_key(|(i, _)| *i)
        {
            let out = self.pending[..idx].to_string();
            self.pending.clear();
            return (out, true);
        }
        // Keep the longest suffix that is a prefix of some stop string.
        let mut keep = 0;
        for s in &self.stops {
            for l in (1..s.len().min(self.pending.len() + 1)).rev() {
                if self.pending.is_char_boundary(self.pending.len() - l)
                    && s.starts_with(&self.pending[self.pending.len() - l..])
                {
                    keep = keep.max(l);
                    break;
                }
            }
        }
        let _ = self.max_len;
        let cut = self.pending.len() - keep;
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (out, false)
    }

    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

/// Prefill chunk sizes that keep attention scratch under ~256 MiB.
pub fn prefill_chunk(cfg: &ModelConfig, pos: usize) -> usize {
    let budget = 256usize << 20;
    let per_token = cfg.num_heads * 4 * (pos + 512);
    (budget / per_token.max(1)).clamp(16, 512)
}

/// Files that make up a model on disk.
#[derive(Clone, Debug)]
pub struct ModelFiles {
    pub dir: PathBuf,
}

impl ModelFiles {
    pub fn new(dir: &Path) -> Result<ModelFiles> {
        if !dir.join("config.json").exists() {
            bail!("{} has no config.json", dir.display());
        }
        if !dir.join("tokenizer.json").exists() {
            bail!(
                "{} has no tokenizer.json (Tendril needs the HuggingFace tokenizer file)",
                dir.display()
            );
        }
        Ok(ModelFiles {
            dir: dir.to_path_buf(),
        })
    }
    pub fn config(&self) -> Result<ModelConfig> {
        ModelConfig::from_file(&self.dir.join("config.json"))
    }
}

pub struct GenerateRequest {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    /// Keep going past end-of-sequence tokens.
    pub ignore_eos: bool,
}

pub enum GenEvent<'a> {
    Text(&'a str),
    Done {
        reason: FinishReason,
        timing: &'a Timing,
    },
}

/// A model with every stage in this process.
pub struct LocalModel {
    pub cfg: Arc<ModelConfig>,
    pub tok: Tok,
    pub stages: Vec<Stage>,
    pub files: ModelFiles,
    next_seq: u64,
}

impl LocalModel {
    pub fn load(
        files: ModelFiles,
        device: Device,
        format: WeightFormat,
        specs: Option<Vec<StageSpec>>,
    ) -> Result<LocalModel> {
        let cfg = Arc::new(files.config()?);
        let tok = Tok::from_dir(&files.dir, &cfg.eos_token_ids, cfg.bos_token_id)?;
        let ws = WeightStore::open_dir(&files.dir)?;
        let dtype = activation_dtype(&device, &cfg.torch_dtype);
        let opts = LoadOptions {
            format,
            device,
            dtype,
        };
        let specs = specs.unwrap_or_else(|| vec![StageSpec::whole(&cfg)]);
        let stages = specs
            .iter()
            .map(|s| Stage::load(cfg.clone(), &ws, *s, &opts))
            .collect::<Result<Vec<_>>>()?;
        Ok(LocalModel {
            cfg,
            tok,
            stages,
            files,
            next_seq: 1,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.stages.iter().map(|s| s.weight_bytes).sum()
    }

    fn step(&mut self, seq: u64, pos: usize, toks: &[u32], want: bool) -> Result<Option<Vec<f32>>> {
        let mut input = StageInput::Tokens(toks.to_vec());
        for s in self.stages.iter_mut() {
            match s.forward(seq, pos, input, want)? {
                StageOutput::Hidden(h) => input = StageInput::Hidden(h),
                StageOutput::Logits(l) => return Ok(Some(l)),
                StageOutput::Nothing => return Ok(None),
            }
        }
        bail!("pipeline has no output head")
    }

    pub fn release(&mut self, seq: u64) {
        for s in self.stages.iter_mut() {
            s.release(seq);
        }
    }

    /// Generate, calling `on` for every piece of text. Return false from `on` to cancel.
    pub fn generate(
        &mut self,
        req: GenerateRequest,
        mut on: impl FnMut(GenEvent) -> bool,
    ) -> Result<Timing> {
        let seq = self.next_seq;
        self.next_seq += 1;
        let t0 = Instant::now();
        let mut timing = Timing {
            prompt_tokens: req.prompt.len(),
            ..Default::default()
        };
        if req.prompt.is_empty() {
            bail!("empty prompt");
        }
        let mut sampler = Sampler::new(req.params.clone(), &req.prompt);
        let mut pos = 0;
        let mut logits = None;
        while pos < req.prompt.len() {
            let n = prefill_chunk(&self.cfg, pos).min(req.prompt.len() - pos);
            let last = pos + n == req.prompt.len();
            let r = self.step(seq, pos, &req.prompt[pos..pos + n], last);
            let r = match r {
                Ok(r) => r,
                Err(e) => {
                    self.release(seq);
                    return Err(e);
                }
            };
            pos += n;
            if last {
                logits = r;
            }
        }
        let mut logits = logits.expect("final prefill chunk returns logits");
        let mut detok = Detokenizer::new();
        let mut stop = StopMatcher::new(req.stop);
        let mut reason = FinishReason::Length;
        for i in 0..req.max_tokens {
            let tok = sampler.sample(&mut logits);
            if i == 0 {
                timing.ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            timing.completion_tokens += 1;
            if self.tok.is_stop(tok) && !req.ignore_eos {
                reason = FinishReason::Stop;
                break;
            }
            let delta = detok.push(&self.tok, tok)?;
            let (emit, stopped) = stop.push(&delta);
            if !emit.is_empty() && !on(GenEvent::Text(&emit)) {
                reason = FinishReason::Cancelled;
                break;
            }
            if stopped {
                reason = FinishReason::Stop;
                break;
            }
            if i + 1 == req.max_tokens || pos + 1 >= self.cfg.max_position {
                break;
            }
            match self.step(seq, pos, &[tok], true)? {
                Some(l) => logits = l,
                None => bail!("no logits"),
            }
            pos += 1;
        }
        let rest = stop.flush();
        if !rest.is_empty() && reason != FinishReason::Stop {
            on(GenEvent::Text(&rest));
        }
        self.release(seq);
        timing.total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        on(GenEvent::Done {
            reason,
            timing: &timing,
        });
        Ok(timing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_matcher_holds_back() {
        let mut s = StopMatcher::new(vec!["<END>".into()]);
        assert_eq!(s.push("hello <E"), ("hello ".to_string(), false));
        assert_eq!(s.push("N"), (String::new(), false));
        assert_eq!(s.push("D> tail"), (String::new(), true));
        let mut s = StopMatcher::new(vec!["<END>".into()]);
        assert_eq!(s.push("a<b"), ("a<b".to_string(), false));
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn generates_from_tiny_model() {
        let dir = tempfile::tempdir().unwrap();
        crate::testing::write_tiny_llama(dir.path(), 3, 64, 1, candle_core::DType::F32).unwrap();
        let mut m = LocalModel::load(
            ModelFiles::new(dir.path()).unwrap(),
            Device::Cpu,
            WeightFormat::Native,
            None,
        )
        .unwrap();
        let prompt = m
            .tok
            .encode_chat(&[crate::tokenizer::ChatMessage::new("user", "hi")])
            .unwrap();
        assert_eq!(prompt[0], 3, "ChatML starts with <|im_start|>");
        let mut text = String::new();
        let t = m
            .generate(
                GenerateRequest {
                    prompt: prompt.clone(),
                    params: SamplingParams::greedy(),
                    max_tokens: 12,
                    stop: vec![],
                    ignore_eos: false,
                },
                |e| {
                    if let GenEvent::Text(s) = e {
                        text.push_str(s);
                    }
                    true
                },
            )
            .unwrap();
        assert!(t.completion_tokens >= 1);
        // Split into two stages: identical greedy output.
        let specs = vec![
            StageSpec {
                layer_start: 0,
                layer_end: 1,
                embed: true,
                head: false,
            },
            StageSpec {
                layer_start: 1,
                layer_end: 3,
                embed: false,
                head: true,
            },
        ];
        let mut m2 = LocalModel::load(
            ModelFiles::new(dir.path()).unwrap(),
            Device::Cpu,
            WeightFormat::Native,
            Some(specs),
        )
        .unwrap();
        let mut text2 = String::new();
        m2.generate(
            GenerateRequest {
                prompt,
                params: SamplingParams::greedy(),
                max_tokens: 12,
                stop: vec![],
                ignore_eos: false,
            },
            |e| {
                if let GenEvent::Text(s) = e {
                    text2.push_str(s);
                }
                true
            },
        )
        .unwrap();
        assert_eq!(text, text2);
        assert_eq!(m.stages[0].active_seqs(), 0, "KV released after generation");
    }
}
