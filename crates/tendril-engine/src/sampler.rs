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
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        }
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
        Sampler {
            params,
            rng,
            counts,
        }
    }

    fn apply_penalties(&self, logits: &mut [f32]) {
        let p = &self.params;
        if p.repetition_penalty == 1.0 && p.presence_penalty == 0.0 && p.frequency_penalty == 0.0 {
            return;
        }
        for (&tok, &c) in &self.counts {
            if let Some(l) = logits.get_mut(tok as usize) {
                if p.repetition_penalty != 1.0 {
                    *l = if *l > 0.0 {
                        *l / p.repetition_penalty
                    } else {
                        *l * p.repetition_penalty
                    };
                }
                *l -= p.presence_penalty + p.frequency_penalty * c as f32;
            }
        }
    }

    /// Pick the next token and remember it.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let d = self.distribution(logits);
        let tok = self.draw(&d);
        self.observe(tok);
        tok
    }

    /// Count a token as part of the history (for penalties).
    pub fn observe(&mut self, tok: u32) {
        *self.counts.entry(tok).or_insert(0) += 1;
    }

    /// The distribution this sampler draws from after penalties, temperature,
    /// top-k, min-p and top-p: `(token, probability)` sorted by probability,
    /// summing to 1. Greedy sampling is a one-hot distribution.
    pub fn distribution(&self, logits: &mut [f32]) -> Vec<(u32, f32)> {
        self.apply_penalties(logits);
        let p = &self.params;
        let argmax = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let max = logits[argmax as usize];
        if p.temperature <= 1e-5 || !max.is_finite() {
            return vec![(argmax, 1.0)];
        }
        let t = p.temperature;
        // Candidates: drop tokens with probability < 1e-7 of the best before sorting.
        let floor = max - t * 16.0;
        let mut cand: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .filter(|(_, &v)| v >= floor)
            .map(|(i, &v)| (i as u32, ((v - max) / t).exp()))
            .collect();
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if p.top_k > 0 && cand.len() > p.top_k {
            cand.truncate(p.top_k);
        }
        let sum: f32 = cand.iter().map(|c| c.1).sum();
        cand.iter_mut().for_each(|c| c.1 /= sum);
        let mut keep = cand.len();
        if p.min_p > 0.0 {
            let thr = cand[0].1 * p.min_p;
            keep = keep.min(
                cand.iter()
                    .position(|c| c.1 < thr)
                    .unwrap_or(cand.len())
                    .max(1),
            );
        }
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            for (i, c) in cand.iter().enumerate().take(keep) {
                acc += c.1;
                if acc >= p.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        cand.truncate(keep);
        let total: f32 = cand.iter().map(|c| c.1).sum();
        cand.iter_mut().for_each(|c| c.1 /= total);
        cand
    }

    /// Draw from a distribution returned by [`Sampler::distribution`].
    pub fn draw(&mut self, dist: &[(u32, f32)]) -> u32 {
        if dist.len() == 1 {
            return dist[0].0;
        }
        let total: f32 = dist.iter().map(|d| d.1).sum();
        let mut r = self.rng.random::<f32>() * total;
        for &(tok, p) in dist {
            r -= p;
            if r <= 0.0 {
                return tok;
            }
        }
        dist.last().map(|d| d.0).unwrap_or(0)
    }

    /// Speculative verification. `rows[i]` are the target logits after
    /// `draft[..i]`; `q[i]` is the (sparse) distribution the draft token
    /// `draft[i]` was drawn from. Returns the accepted draft tokens followed by
    /// exactly one token drawn from the target: the output has the same
    /// distribution as sampling the target one token at a time.
    pub fn verify(
        &mut self,
        rows: &mut [Vec<f32>],
        draft: &[u32],
        q: &[Vec<(u32, f32)>],
    ) -> Vec<u32> {
        let mut out = Vec::with_capacity(draft.len() + 1);
        for (i, &x) in draft.iter().enumerate() {
            let p = self.distribution(&mut rows[i]);
            let px = p.iter().find(|d| d.0 == x).map_or(0.0, |d| d.1);
            let qx = q
                .get(i)
                .and_then(|qi| qi.iter().find(|d| d.0 == x))
                .map_or(0.0, |d| d.1);
            let accept = if qx <= 0.0 {
                px > 0.0 && self.rng.random::<f32>() < px
            } else {
                self.rng.random::<f32>() < (px / qx).min(1.0)
            };
            if accept {
                self.observe(x);
                out.push(x);
                continue;
            }
            // Rejected: draw from the residual max(0, p - q).
            let qi = q.get(i).cloned().unwrap_or_default();
            let residual: Vec<(u32, f32)> = p
                .iter()
                .map(|&(t, pt)| {
                    (
                        t,
                        (pt - qi.iter().find(|d| d.0 == t).map_or(0.0, |d| d.1)).max(0.0),
                    )
                })
                .filter(|d| d.1 > 0.0)
                .collect();
            let tok = if residual.is_empty() {
                self.draw(&p)
            } else {
                self.draw(&residual)
            };
            self.observe(tok);
            out.push(tok);
            return out;
        }
        // Every draft accepted: one bonus token from the last row.
        let p = self.distribution(&mut rows[draft.len()]);
        let tok = self.draw(&p);
        self.observe(tok);
        out.push(tok);
        out
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
        let p = SamplingParams {
            temperature: 1.0,
            seed: Some(42),
            ..Default::default()
        };
        let a: Vec<u32> = {
            let mut s = Sampler::new(p.clone(), &[]);
            (0..20)
                .map(|_| s.sample(&mut [1.0, 1.1, 0.9, 1.05]))
                .collect()
        };
        let b: Vec<u32> = {
            let mut s = Sampler::new(p, &[]);
            (0..20)
                .map(|_| s.sample(&mut [1.0, 1.1, 0.9, 1.05]))
                .collect()
        };
        assert_eq!(a, b);
        assert!(a.iter().any(|&x| x != a[0]), "sampling should vary");
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 2.0,
                top_k: 1,
                ..Default::default()
            },
            &[],
        );
        for _ in 0..10 {
            assert_eq!(s.sample(&mut [0.0, 0.5, 5.0, 1.0]), 2);
        }
    }

    #[test]
    fn greedy_verification_accepts_matching_prefix() {
        let mut s = Sampler::new(SamplingParams::greedy(), &[]);
        // Target argmaxes: 2, 1, 0, 3
        let mut rows = vec![
            vec![0.0, 0.0, 5.0, 0.0],
            vec![0.0, 5.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 5.0],
        ];
        let one = |t: u32| vec![(t, 1.0)];
        // Drafts 2, 1, 3: third is wrong -> [2, 1, 0]
        assert_eq!(
            s.verify(&mut rows.clone(), &[2, 1, 3], &[one(2), one(1), one(3)]),
            vec![2, 1, 0]
        );
        // All right -> bonus token 3.
        assert_eq!(
            s.verify(&mut rows, &[2, 1, 0], &[one(2), one(1), one(0)]),
            vec![2, 1, 0, 3]
        );
    }

    #[test]
    fn sampled_verification_preserves_the_target_distribution() {
        // Target p = [0.6, 0.3, 0.1]; draft always proposes token 1 (one-hot q).
        // Accepted tokens + residual draws must still follow p.
        let logits = [0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            seed: Some(7),
            ..Default::default()
        };
        let mut s = Sampler::new(params, &[]);
        let mut counts = [0usize; 3];
        let n = 60_000;
        for _ in 0..n {
            let mut rows = vec![logits.to_vec(), logits.to_vec()];
            let out = s.verify(&mut rows, &[1], &[vec![(1, 1.0)]]);
            counts[out[0] as usize] += 1;
            s.counts.clear();
        }
        for (i, want) in [0.6, 0.3, 0.1].iter().enumerate() {
            let got = counts[i] as f64 / n as f64;
            assert!((got - want).abs() < 0.01, "token {i}: {got} vs {want}");
        }
    }

    #[test]
    fn repetition_penalty() {
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 0.0,
                repetition_penalty: 10.0,
                ..Default::default()
            },
            &[1],
        );
        assert_eq!(s.sample(&mut [0.0, 2.0, 1.5]), 2);
    }
}
