//! Token sampling: greedy, temperature, top-k, top-p, min-p and penalties.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SamplingParams {
    /// 0 = greedy.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 = disabled.
    pub top_k: usize,
    pub min_p: f32,
    /// 1.0 = disabled (multiplicative, as in HF/llama.cpp).
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        SamplingParams { temperature: 0.0, top_p: 1.0, ..Default::default() }
    }
}

/// Per-sequence sampler state.
pub struct Sampler {
    pub params: SamplingParams,
    rng: StdRng,
    counts: HashMap<u32, u32>,
}

impl Sampler {
    pub fn new(params: SamplingParams, history: &[u32]) -> Sampler {
        let rng = match params.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        let mut counts = HashMap::new();
        for &t in history {
            *counts.entry(t).or_insert(0) += 1;
        }
        Sampler { params, rng, counts }
    }

    fn apply_penalties(&self, logits: &mut [f32]) {
        let p = &self.params;
        if p.repetition_penalty == 1.0 && p.presence_penalty == 0.0 && p.frequency_penalty == 0.0 {
            return;
        }
        for (&tok, &c) in &self.counts {
            if let Some(l) = logits.get_mut(tok as usize) {
                if p.repetition_penalty != 1.0 {
                    *l = if *l > 0.0 { *l / p.repetition_penalty } else { *l * p.repetition_penalty };
                }
                *l -= p.presence_penalty + p.frequency_penalty * c as f32;
            }
        }
    }

    /// Pick the next token and remember it.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        self.apply_penalties(logits);
        let tok = self.pick(logits);
        *self.counts.entry(tok).or_insert(0) += 1;
        tok
    }

    fn pick(&mut self, logits: &[f32]) -> u32 {
        let p = &self.params;
        let argmax = || logits.iter().enumerate().filter(|(_, v)| !v.is_nan()).max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
        if p.temperature <= 1e-5 {
            return argmax();
        }
        let t = p.temperature;
        let max = logits.iter().cloned().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return argmax();
        }
        // Candidates: drop tokens with probability < 1e-7 of the best before sorting.
        let floor = max - t * 16.0;
        let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().filter(|(_, &v)| v >= floor).map(|(i, &v)| (i as u32, (v - max) / t)).collect();
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if p.top_k > 0 && cand.len() > p.top_k {
            cand.truncate(p.top_k);
        }
        let mut probs: Vec<f32> = cand.iter().map(|(_, l)| l.exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= sum);
        let mut keep = probs.len();
        if p.min_p > 0.0 {
            let thr = probs[0] * p.min_p;
            keep = keep.min(probs.iter().position(|&x| x < thr).unwrap_or(probs.len()).max(1));
        }
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            for (i, &x) in probs.iter().enumerate().take(keep) {
                acc += x;
                if acc >= p.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        let total: f32 = probs[..keep].iter().sum();
        let mut r = self.rng.random::<f32>() * total;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return cand[i].0;
            }
        }
        cand[keep - 1].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_seeded() {
        let mut s = Sampler::new(SamplingParams::greedy(), &[]);
        let mut l = vec![0.1, 3.0, 0.2];
        assert_eq!(s.sample(&mut l), 1);
        let p = SamplingParams { temperature: 1.0, seed: Some(42), ..Default::default() };
        let a: Vec<u32> = {
            let mut s = Sampler::new(p.clone(), &[]);
            (0..20).map(|_| s.sample(&mut vec![1.0, 1.1, 0.9, 1.05])).collect()
        };
        let b: Vec<u32> = {
            let mut s = Sampler::new(p, &[]);
            (0..20).map(|_| s.sample(&mut vec![1.0, 1.1, 0.9, 1.05])).collect()
        };
        assert_eq!(a, b);
        assert!(a.iter().any(|&x| x != a[0]), "sampling should vary");
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(SamplingParams { temperature: 2.0, top_k: 1, ..Default::default() }, &[]);
        for _ in 0..10 {
            assert_eq!(s.sample(&mut vec![0.0, 0.5, 5.0, 1.0]), 2);
        }
    }

    #[test]
    fn repetition_penalty() {
        let mut s = Sampler::new(SamplingParams { temperature: 0.0, repetition_penalty: 10.0, ..Default::default() }, &[1]);
        assert_eq!(s.sample(&mut vec![0.0, 2.0, 1.5]), 2);
    }
}
