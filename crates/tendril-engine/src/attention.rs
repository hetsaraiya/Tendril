//! Streaming attention for the CPU backend.
//!
//! Reads keys and values straight from the KV cache buffers (no per-step
//! copies or transposes) and never materializes a `queries × keys` score
//! matrix: each (head, query) row is computed independently across all cores.

use crate::cpu_kernels::dot;
use anyhow::{bail, Result};
use candle_core::{CpuStorage, Storage, Tensor};

pub struct AttnParams {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Number of queries.
    pub seq: usize,
    /// Absolute position of the first query.
    pub pos: usize,
    /// Absolute position of cache entry 0.
    pub key_offset: usize,
    /// Valid cache entries.
    pub len: usize,
    /// Capacity of the cache buffers (entries per head).
    pub cap: usize,
    pub scale: f32,
    pub softcap: Option<f32>,
    pub window: Option<usize>,
}

/// Run `f` with the f32 slice behind a contiguous CPU tensor.
fn with_f32<R>(t: &Tensor, f: impl FnOnce(&[f32]) -> R) -> Result<R> {
    let (st, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        bail!("attention input must be contiguous");
    }
    match &*st {
        Storage::Cpu(CpuStorage::F32(v)) => Ok(f(&v[layout.start_offset()..])),
        _ => bail!("CPU f32 tensor expected"),
    }
}

/// q: [1, heads, seq, hd]; k, v: full cache buffers [1, kv_heads, cap, hd].
/// Returns [seq, heads * hd] ready for the output projection.
pub fn attend(q: &Tensor, k: &Tensor, v: &Tensor, p: &AttnParams) -> Result<Vec<f32>> {
    let (nh, nkv, hd, seq) = (p.heads, p.kv_heads, p.head_dim, p.seq);
    let groups = nh / nkv;
    let mut out = vec![0f32; seq * nh * hd];
    struct Ptr(*mut f32);
    unsafe impl Sync for Ptr {}
    let op = Ptr(out.as_mut_ptr());
    let op = &op;
    with_f32(q, |q| {
        with_f32(k, |k| {
            with_f32(v, |v| {
                crate::pool::global().run(nh * seq, &|task| {
                    let h = task / seq;
                    let i = task % seq;
                    let g = h / groups;
                    let qrow = &q[(h * seq + i) * hd..(h * seq + i + 1) * hd];
                    let qp = p.pos + i;
                    let hi = (qp + 1).saturating_sub(p.key_offset).min(p.len);
                    let lo = match p.window {
                        Some(w) => (qp + 1)
                            .saturating_sub(w)
                            .saturating_sub(p.key_offset)
                            .min(hi),
                        None => 0,
                    };
                    let kbase = g * p.cap * hd;
                    let mut scores = Vec::with_capacity(hi - lo);
                    let mut max = f32::NEG_INFINITY;
                    for j in lo..hi {
                        let mut s = dot(qrow, &k[kbase + j * hd..kbase + (j + 1) * hd]) * p.scale;
                        if let Some(c) = p.softcap {
                            s = (s / c).tanh() * c;
                        }
                        max = max.max(s);
                        scores.push(s);
                    }
                    let mut sum = 0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    // SAFETY: each task writes its own disjoint [hd] row.
                    let o =
                        unsafe { std::slice::from_raw_parts_mut(op.0.add((i * nh + h) * hd), hd) };
                    for (jj, j) in (lo..hi).enumerate() {
                        let w = scores[jj] * inv;
                        let vrow = &v[kbase + j * hd..kbase + (j + 1) * hd];
                        for d in 0..hd {
                            o[d] += w * vrow[d];
                        }
                    }
                });
            })
        })
    })???;
    Ok(out)
}
