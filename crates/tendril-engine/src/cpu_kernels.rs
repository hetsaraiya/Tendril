//! Hand-written CPU matrix kernels for the decode hot path.
//!
//! Token-by-token decoding multiplies a handful of activation rows by every
//! weight matrix, so it is bound by how fast weights stream from memory.
//! These kernels read weights in their stored precision (bf16/f16/q8_0/f32),
//! widen to f32 in registers, and split rows across all cores. On a 4-core
//! Xeon they run 3–6× faster than the generic matmul path for batch-1 decode.

use half::f16;
use rayon::prelude::*;
use std::sync::OnceLock;

pub const Q8_BLOCK: usize = 32;

/// One q8_0 block: 32 signed bytes sharing an f16 scale.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ8 {
    pub d: u16,
    pub qs: [i8; Q8_BLOCK],
}

/// Weights of an `n × k` matrix (row-major, one row per output feature).
pub enum CpuWeights {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
    F16(Vec<u16>),
    Q8(Vec<BlockQ8>),
}

impl CpuWeights {
    pub fn bytes(&self) -> usize {
        match self {
            CpuWeights::F32(v) => v.len() * 4,
            CpuWeights::Bf16(v) | CpuWeights::F16(v) => v.len() * 2,
            CpuWeights::Q8(v) => v.len() * std::mem::size_of::<BlockQ8>(),
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            CpuWeights::F32(_) => "f32",
            CpuWeights::Bf16(_) => "bf16",
            CpuWeights::F16(_) => "f16",
            CpuWeights::Q8(_) => "q8_0",
        }
    }
}

fn f16_table() -> &'static [f32] {
    static T: OnceLock<Vec<f32>> = OnceLock::new();
    T.get_or_init(|| (0..=u16::MAX).map(|b| f16::from_bits(b).to_f32()).collect())
}

/// Branch-free f16 → f32 that LLVM can vectorize (handles subnormals, inf, NaN).
#[inline(always)]
fn f16_to_f32(b: u16) -> f32 {
    let h = b as u32;
    let sign = (h & 0x8000) << 16;
    let em = h & 0x7fff;
    // Rebias by multiplying with 2^112; exact for normals and subnormals.
    let mag = f32::from_bits(em << 13) * f32::from_bits(0x7780_0000);
    let mut bits = mag.to_bits();
    if em >= 0x7c00 {
        bits |= 0x7f80_0000; // inf / NaN
    }
    f32::from_bits(bits | sign)
}

#[inline(always)]
fn dot_f16(w: &[u16], x: &[f32]) -> f32 {
    let mut acc = [0f32; L];
    let chunks = w.len() / L;
    for c in 0..chunks {
        let wo = &w[c * L..c * L + L];
        let xo = &x[c * L..c * L + L];
        for i in 0..L {
            acc[i] += f16_to_f32(wo[i]) * xo[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * L..w.len() {
        s += f16_to_f32(w[i]) * x[i];
    }
    s
}

/// Activations quantized to q8 blocks for integer dot products.
pub struct QuantX {
    pub d: Vec<f32>,
    pub q: Vec<i8>,
}

pub fn quantize_x(x: &[f32]) -> QuantX {
    let nb = x.len() / Q8_BLOCK;
    let mut d = vec![0f32; nb];
    let mut q = vec![0i8; x.len()];
    for b in 0..nb {
        let blk = &x[b * Q8_BLOCK..b * Q8_BLOCK + Q8_BLOCK];
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let dd = amax / 127.0;
        let id = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        for i in 0..Q8_BLOCK {
            q[b * Q8_BLOCK + i] = (blk[i] * id).round() as i8;
        }
        d[b] = dd;
    }
    QuantX { d, q }
}

#[inline(always)]
fn dot_q8q8(w: &[BlockQ8], xq: &QuantX, t: &[f32]) -> f32 {
    let mut total = 0f32;
    for (b, blk) in w.iter().enumerate() {
        let xo = &xq.q[b * Q8_BLOCK..b * Q8_BLOCK + Q8_BLOCK];
        let mut acc = 0i32;
        for i in 0..Q8_BLOCK {
            acc += blk.qs[i] as i16 as i32 * xo[i] as i16 as i32;
        }
        total += t[blk.d as usize] * xq.d[b] * acc as f32;
    }
    total
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{BlockQ8, QuantX, Q8_BLOCK};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// q8_0 · q8 activations with the maddubs sign trick (as in llama.cpp).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8q8(w: &[BlockQ8], xq: &QuantX, t: &[f32]) -> f32 {
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        for (b, blk) in w.iter().enumerate() {
            let qw = _mm256_loadu_si256(blk.qs.as_ptr() as *const __m256i);
            let qx = _mm256_loadu_si256(xq.q.as_ptr().add(b * Q8_BLOCK) as *const __m256i);
            let ax = _mm256_sign_epi8(qw, qw);
            let sy = _mm256_sign_epi8(qx, qw);
            let d16 = _mm256_maddubs_epi16(ax, sy);
            let d32 = _mm256_madd_epi16(d16, ones);
            let scale = _mm256_set1_ps(*t.get_unchecked(blk.d as usize) * *xq.d.get_unchecked(b));
            acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(d32), acc);
        }
        hsum(acc)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len() / 16 * 16;
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0;
        while i < n {
            let h0 = _mm_loadu_si128(w.as_ptr().add(i) as *const __m128i);
            let h1 = _mm_loadu_si128(w.as_ptr().add(i + 8) as *const __m128i);
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(h0), _mm256_loadu_ps(x.as_ptr().add(i)), a0);
            a1 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(h1),
                _mm256_loadu_ps(x.as_ptr().add(i + 8)),
                a1,
            );
            i += 16;
        }
        let mut s = hsum(_mm256_add_ps(a0, a1));
        for j in n..w.len() {
            s += super::f16_to_f32(w[j]) * x[j];
        }
        s
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len() / 16 * 16;
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0;
        while i < n {
            let h0 = _mm_loadu_si128(w.as_ptr().add(i) as *const __m128i);
            let h1 = _mm_loadu_si128(w.as_ptr().add(i + 8) as *const __m128i);
            let f0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16));
            let f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
            a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x.as_ptr().add(i)), a0);
            a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x.as_ptr().add(i + 8)), a1);
            i += 16;
        }
        let mut s = hsum(_mm256_add_ps(a0, a1));
        for j in n..w.len() {
            s += super::bf16_to_f32(w[j]) * x[j];
        }
        s
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(w: &[f32], x: &[f32]) -> f32 {
        let n = w.len() / 32 * 32;
        let mut a = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < n {
            for (l, acc) in a.iter_mut().enumerate() {
                *acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(w.as_ptr().add(i + l * 8)),
                    _mm256_loadu_ps(x.as_ptr().add(i + l * 8)),
                    *acc,
                );
            }
            i += 32;
        }
        let mut s = hsum(_mm256_add_ps(
            _mm256_add_ps(a[0], a[1]),
            _mm256_add_ps(a[2], a[3]),
        ));
        for j in n..w.len() {
            s += w[j] * x[j];
        }
        s
    }
}

#[inline(always)]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Quantize a row-major f32 matrix to q8_0 (k must be a multiple of 32).
pub fn quantize_q8(src: &[f32]) -> Vec<BlockQ8> {
    src.par_chunks(Q8_BLOCK)
        .map(|blk| {
            let amax = blk.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut qs = [0i8; Q8_BLOCK];
            for (q, &x) in qs.iter_mut().zip(blk) {
                *q = (x * id).round().clamp(-127.0, 127.0) as i8;
            }
            BlockQ8 {
                d: f16::from_f32(d).to_bits(),
                qs,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Inner products. Written so LLVM vectorizes them: fixed-width lanes with
// independent accumulators. Compiled twice on x86 (baseline and AVX2+FMA).

const L: usize = 16;

#[inline(always)]
fn dot_f32(w: &[f32], x: &[f32]) -> f32 {
    let mut acc = [0f32; L];
    let chunks = w.len() / L;
    for c in 0..chunks {
        let wo = &w[c * L..c * L + L];
        let xo = &x[c * L..c * L + L];
        for i in 0..L {
            acc[i] += wo[i] * xo[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * L..w.len() {
        s += w[i] * x[i];
    }
    s
}

#[inline(always)]
fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    let mut acc = [0f32; L];
    let chunks = w.len() / L;
    for c in 0..chunks {
        let wo = &w[c * L..c * L + L];
        let xo = &x[c * L..c * L + L];
        for i in 0..L {
            acc[i] += bf16_to_f32(wo[i]) * xo[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * L..w.len() {
        s += bf16_to_f32(w[i]) * x[i];
    }
    s
}

#[inline(always)]
fn dot_q8(w: &[BlockQ8], x: &[f32]) -> f32 {
    let t = f16_table();
    let mut total = 0f32;
    for (b, blk) in w.iter().enumerate() {
        let xo = &x[b * Q8_BLOCK..b * Q8_BLOCK + Q8_BLOCK];
        let mut acc = [0f32; L];
        for h in 0..Q8_BLOCK / L {
            for i in 0..L {
                acc[i] += blk.qs[h * L + i] as f32 * xo[h * L + i];
            }
        }
        total += t[blk.d as usize] * acc.iter().sum::<f32>();
    }
    total
}

/// Widen one weight row into `buf` (f32).
#[inline(always)]
fn widen_row(w: &CpuWeights, row: usize, k: usize, buf: &mut [f32]) {
    match w {
        CpuWeights::F32(v) => buf.copy_from_slice(&v[row * k..row * k + k]),
        CpuWeights::Bf16(v) => {
            for (o, &b) in buf.iter_mut().zip(&v[row * k..row * k + k]) {
                *o = bf16_to_f32(b);
            }
        }
        CpuWeights::F16(v) => {
            for (o, &b) in buf.iter_mut().zip(&v[row * k..row * k + k]) {
                *o = f16_to_f32(b);
            }
        }
        CpuWeights::Q8(v) => {
            let t = f16_table();
            let bpr = k / Q8_BLOCK;
            for (bi, blk) in v[row * bpr..row * bpr + bpr].iter().enumerate() {
                let d = t[blk.d as usize];
                for i in 0..Q8_BLOCK {
                    buf[bi * Q8_BLOCK + i] = blk.qs[i] as f32 * d;
                }
            }
        }
    }
}

/// Compute rows [r0, r1) of the output for all `m` inputs.
/// `out` is laid out as `m × n`; this writes a column band.
#[inline(always)]
fn band(
    x: &[f32],
    xq: Option<&QuantX>,
    m: usize,
    k: usize,
    n: usize,
    w: &CpuWeights,
    r0: usize,
    r1: usize,
    out_band: &mut [f32],
) {
    // out_band: m × (r1 - r0), transposed back by the caller.
    let width = r1 - r0;
    if m == 1 {
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            let t = f16_table();
            for (j, o) in (r0..r1).zip(out_band.iter_mut()) {
                // SAFETY: AVX2/FMA/F16C presence checked at runtime.
                *o = unsafe {
                    match w {
                        CpuWeights::F32(v) => x86::dot_f32(&v[j * k..j * k + k], x),
                        CpuWeights::Bf16(v) => x86::dot_bf16(&v[j * k..j * k + k], x),
                        CpuWeights::F16(v) => x86::dot_f16(&v[j * k..j * k + k], x),
                        CpuWeights::Q8(v) => {
                            let bpr = k / Q8_BLOCK;
                            match xq {
                                Some(xq) => x86::dot_q8q8(&v[j * bpr..j * bpr + bpr], xq, t),
                                None => dot_q8(&v[j * bpr..j * bpr + bpr], x),
                            }
                        }
                    }
                };
            }
            return;
        }
        for (j, o) in (r0..r1).zip(out_band.iter_mut()) {
            *o = match w {
                CpuWeights::F32(v) => dot_f32(&v[j * k..j * k + k], x),
                CpuWeights::Bf16(v) => dot_bf16(&v[j * k..j * k + k], x),
                CpuWeights::F16(v) => dot_f16(&v[j * k..j * k + k], x),
                CpuWeights::Q8(v) => {
                    let bpr = k / Q8_BLOCK;
                    match xq {
                        Some(xq) => dot_q8q8(&v[j * bpr..j * bpr + bpr], xq, f16_table()),
                        None => dot_q8(&v[j * bpr..j * bpr + bpr], x),
                    }
                }
            };
        }
        return;
    }
    let _ = n;
    let mut buf = vec![0f32; k];
    for (jj, j) in (r0..r1).enumerate() {
        widen_row(w, j, k, &mut buf);
        for r in 0..m {
            #[cfg(target_arch = "x86_64")]
            if has_avx2() {
                // SAFETY: feature checked at runtime.
                out_band[r * width + jj] = unsafe { x86::dot_f32(&buf, &x[r * k..r * k + k]) };
                continue;
            }
            out_band[r * width + jj] = dot_f32(&buf, &x[r * k..r * k + k]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn band_avx2(
    x: &[f32],
    xq: Option<&QuantX>,
    m: usize,
    k: usize,
    n: usize,
    w: &CpuWeights,
    r0: usize,
    r1: usize,
    out: &mut [f32],
) {
    band(x, xq, m, k, n, w, r0, r1, out)
}

fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static F: OnceLock<bool> = OnceLock::new();
        *F.get_or_init(|| {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn run_band(
    x: &[f32],
    xq: Option<&QuantX>,
    m: usize,
    k: usize,
    n: usize,
    w: &CpuWeights,
    r0: usize,
    r1: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: guarded by runtime feature detection.
        unsafe { band_avx2(x, xq, m, k, n, w, r0, r1, out) };
        return;
    }
    let _ = has_avx2;
    band(x, xq, m, k, n, w, r0, r1, out)
}

/// `out[m × n] = x[m × k] · Wᵀ` where W is `n × k`.
pub fn matmul(x: &[f32], m: usize, k: usize, w: &CpuWeights, n: usize, out: &mut [f32]) {
    assert_eq!(x.len(), m * k, "input shape");
    assert_eq!(out.len(), m * n, "output shape");
    // Bands of output features; small enough to balance, large enough to amortize.
    let pool = crate::pool::global();
    let threads = pool.threads();
    let band_rows = (n / (threads * 6)).clamp(16, 512);
    let nb = n.div_ceil(band_rows);
    let xq = (m == 1 && matches!(w, CpuWeights::Q8(_))).then(|| quantize_x(x));
    let xq = xq.as_ref();
    struct OutPtr(*mut f32);
    // SAFETY: bands write disjoint regions of `out`.
    unsafe impl Sync for OutPtr {}
    let op = OutPtr(out.as_mut_ptr());
    let op = &op;
    if m == 1 {
        pool.run(nb, &|b| {
            let s = b * band_rows;
            let e = (s + band_rows).min(n);
            let o = unsafe { std::slice::from_raw_parts_mut(op.0.add(s), e - s) };
            run_band(x, xq, 1, k, n, w, s, e, o);
        });
        return;
    }
    pool.run(nb, &|b| {
        let s = b * band_rows;
        let e = (s + band_rows).min(n);
        let wdt = e - s;
        let mut o = vec![0f32; m * wdt];
        run_band(x, None, m, k, n, w, s, e, &mut o);
        for r in 0..m {
            let dst = unsafe { std::slice::from_raw_parts_mut(op.0.add(r * n + s), wdt) };
            dst.copy_from_slice(&o[r * wdt..r * wdt + wdt]);
        }
    });
}

/// f32 dot product with the best available SIMD.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: feature checked at runtime.
        return unsafe { x86::dot_f32(a, b) };
    }
    dot_f32(a, b)
}

/// Widen one row to f32 (embedding lookup).
pub fn row_f32(w: &CpuWeights, row: usize, k: usize, out: &mut [f32]) {
    widen_row(w, row, k, out)
}

/// Widen rows [s, e) to f32.
pub fn rows_f32(w: &CpuWeights, s: usize, e: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; (e - s) * k];
    out.par_chunks_mut(k)
        .enumerate()
        .for_each(|(j, row)| widen_row(w, s + j, k, row));
    out
}

/// Dequantize/widen all rows to f32 (used for large prefill batches).
pub fn to_f32(w: &CpuWeights, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * k];
    out.par_chunks_mut(k)
        .enumerate()
        .for_each(|(j, row)| widen_row(w, j, k, row));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
        let mut o = vec![0f32; m * n];
        for r in 0..m {
            for j in 0..n {
                o[r * n + j] = (0..k).map(|i| x[r * k + i] * w[j * k + i]).sum();
            }
        }
        o
    }

    fn rnd(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn all_formats_match_reference() {
        let (m, k, n) = (3, 96, 70);
        let x = rnd(m * k, 1);
        let wf = rnd(n * k, 2);
        let exp = reference(&x, m, k, &wf, n);
        let bf: Vec<u16> = wf
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_bits())
            .collect();
        let hf: Vec<u16> = wf.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        for (w, tol) in [
            (CpuWeights::F32(wf.clone()), 1e-4),
            (CpuWeights::Bf16(bf), 2e-2),
            (CpuWeights::F16(hf), 2e-3),
            (CpuWeights::Q8(quantize_q8(&wf)), 3e-2),
        ] {
            for mm in [1, m] {
                let mut out = vec![0f32; mm * n];
                matmul(&x[..mm * k], mm, k, &w, n, &mut out);
                for (a, b) in out.iter().zip(&exp[..mm * n]) {
                    assert!((a - b).abs() < tol, "{} m={mm}: {a} vs {b}", w.label());
                }
            }
            let full = to_f32(&w, n, k);
            assert!((full[5] - wf[5]).abs() < 0.01);
        }
    }
}
