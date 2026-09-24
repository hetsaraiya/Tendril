//! Linear layers with the fastest available implementation per device.

use crate::cpu_kernels::{self, quantize_q8, CpuWeights, Q8_BLOCK};
use crate::weights::WeightStore;
use anyhow::{bail, Result};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use std::sync::Arc;

/// Precision requested for weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    /// Keep the checkpoint's precision.
    Native,
    Q8_0,
    Q4K,
    Q6K,
}

impl WeightFormat {
    pub fn parse(s: &str) -> Option<WeightFormat> {
        Some(match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "" | "native" | "none" | "bf16" | "f16" | "fp16" => WeightFormat::Native,
            "q8" | "q8_0" | "int8" => WeightFormat::Q8_0,
            "q4" | "q4_k" | "q4_k_m" => WeightFormat::Q4K,
            "q6" | "q6_k" => WeightFormat::Q6K,
            _ => return None,
        })
    }
    pub fn label(self) -> &'static str {
        match self {
            WeightFormat::Native => "native",
            WeightFormat::Q8_0 => "q8_0",
            WeightFormat::Q4K => "q4_k",
            WeightFormat::Q6K => "q6_k",
        }
    }
}

enum Kind {
    /// Tendril's CPU kernels.
    Cpu(CpuWeights),
    /// candle dense matmul (GPU).
    Dense(Tensor),
    /// candle quantized matmul (any device).
    Quant(QMatMul),
}

pub struct Linear {
    kind: Kind,
    bias: Option<Tensor>,
    pub out_features: usize,
    pub in_features: usize,
}

/// Rows at or above this use a widened f32 GEMM instead of the multi-row
/// kernel (which reads each weight once for all rows; faster below ~96 rows).
const GEMM_ROWS: usize = 96;

impl Linear {
    /// Load `name` (+ optional bias) from the store.
    pub fn load(
        ws: &WeightStore,
        name: &str,
        bias: Option<&str>,
        device: &Device,
        act: DType,
        fmt: WeightFormat,
    ) -> Result<Linear> {
        let (dt, shape, data) = ws.raw(name)?;
        if shape.len() != 2 {
            bail!("{name}: expected a 2-D weight, got {shape:?}");
        }
        let (n, k) = (shape[0], shape[1]);
        // GPUs keep native precision: load straight to the device, no f32 detour.
        let kind = if !device.is_cpu() && fmt == WeightFormat::Native {
            Kind::Dense(ws.tensor(name, device, act)?)
        } else {
            Self::make_kind(Some((dt, data)), None, n, k, device, act, fmt, || {
                ws.tensor(name, &Device::Cpu, DType::F32)
            })?
        };
        let bias = match bias {
            Some(b) if ws.has(b) => Some(ws.tensor(b, device, act)?),
            _ => None,
        };
        Ok(Linear {
            kind,
            bias,
            out_features: n,
            in_features: k,
        })
    }

    /// Build from an f32 CPU tensor (used when splitting fused projections).
    pub fn from_f32(
        w: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        act: DType,
        fmt: WeightFormat,
        native: safetensors::Dtype,
    ) -> Result<Linear> {
        let (n, k) = w.dims2()?;
        let kind = Self::make_kind(None, Some(native), n, k, device, act, fmt, || Ok(w.clone()))?;
        Ok(Linear {
            kind,
            bias,
            out_features: n,
            in_features: k,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn make_kind(
        raw: Option<(safetensors::Dtype, &[u8])>,
        native_hint: Option<safetensors::Dtype>,
        n: usize,
        k: usize,
        device: &Device,
        act: DType,
        fmt: WeightFormat,
        f32_tensor: impl Fn() -> Result<Tensor>,
    ) -> Result<Kind> {
        use safetensors::Dtype as S;
        let quant = match fmt {
            WeightFormat::Native => None,
            WeightFormat::Q8_0 => Some(GgmlDType::Q8_0),
            WeightFormat::Q4K => Some(GgmlDType::Q4K),
            WeightFormat::Q6K => Some(GgmlDType::Q6K),
        };
        if device.is_cpu() {
            if fmt == WeightFormat::Q8_0 && k.is_multiple_of(Q8_BLOCK) {
                let w: Vec<f32> = f32_tensor()?.flatten_all()?.to_vec1()?;
                return Ok(Kind::Cpu(CpuWeights::Q8(quantize_q8(&w))));
            }
            if let Some(q) = quant {
                if k.is_multiple_of(q.block_size()) {
                    let qt = QTensor::quantize(&f32_tensor()?, q)?;
                    return Ok(Kind::Quant(QMatMul::from_qtensor(qt)?));
                }
            }
            let dtype = raw.map(|r| r.0).or(native_hint).unwrap_or(S::F32);
            let as_u16 = |b: &[u8]| -> Vec<u16> {
                b.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            };
            return Ok(Kind::Cpu(match (dtype, raw) {
                (S::BF16, Some((_, b))) => CpuWeights::Bf16(as_u16(b)),
                (S::F16, Some((_, b))) => CpuWeights::F16(as_u16(b)),
                (S::BF16, None) => {
                    let v: Vec<u16> = f32_tensor()?
                        .to_dtype(DType::BF16)?
                        .flatten_all()?
                        .to_vec1::<half::bf16>()?
                        .into_iter()
                        .map(|x| x.to_bits())
                        .collect();
                    CpuWeights::Bf16(v)
                }
                (S::F16, None) => {
                    let v: Vec<u16> = f32_tensor()?
                        .to_dtype(DType::F16)?
                        .flatten_all()?
                        .to_vec1::<half::f16>()?
                        .into_iter()
                        .map(|x| x.to_bits())
                        .collect();
                    CpuWeights::F16(v)
                }
                _ => CpuWeights::F32(f32_tensor()?.flatten_all()?.to_vec1()?),
            }));
        }
        if let Some(q) = quant {
            if k.is_multiple_of(q.block_size()) {
                let qt = QTensor::quantize_onto(&f32_tensor()?, q, device)?;
                return Ok(Kind::Quant(QMatMul::from_qtensor(qt)?));
            }
        }
        let t = f32_tensor()?.to_dtype(act)?.to_device(device)?;
        let _ = n;
        Ok(Kind::Dense(t))
    }

    /// Stack several linears with the same input into one (rows concatenated),
    /// so one pass over the input serves all of them. Returns None when the
    /// representations cannot be fused (e.g. candle-quantized tensors).
    pub fn fuse(parts: Vec<Linear>) -> std::result::Result<Linear, Vec<Linear>> {
        let k = parts[0].in_features;
        if parts.iter().any(|p| p.in_features != k) {
            return Err(parts);
        }
        let n: usize = parts.iter().map(|p| p.out_features).sum();
        let all_cpu_same = parts.iter().all(|p| matches!(&p.kind, Kind::Cpu(w) if std::mem::discriminant(w) == match &parts[0].kind { Kind::Cpu(w0) => std::mem::discriminant(w0), _ => unreachable!() }));
        let all_dense = parts.iter().all(|p| matches!(p.kind, Kind::Dense(_)));
        let bias = if parts.iter().all(|p| p.bias.is_some()) {
            let bs: Vec<Tensor> = parts.iter().map(|p| p.bias.clone().unwrap()).collect();
            match Tensor::cat(&bs, 0) {
                Ok(b) => Some(b),
                Err(_) => return Err(parts),
            }
        } else if parts.iter().any(|p| p.bias.is_some()) {
            return Err(parts);
        } else {
            None
        };
        if matches!(parts[0].kind, Kind::Cpu(_)) && all_cpu_same {
            let mut it = parts.into_iter().map(|p| match p.kind {
                Kind::Cpu(w) => w,
                _ => unreachable!(),
            });
            let first = it.next().unwrap();
            let fused = it.fold(first, |acc, w| match (acc, w) {
                (CpuWeights::F32(mut a), CpuWeights::F32(b)) => {
                    a.extend(b);
                    CpuWeights::F32(a)
                }
                (CpuWeights::Bf16(mut a), CpuWeights::Bf16(b)) => {
                    a.extend(b);
                    CpuWeights::Bf16(a)
                }
                (CpuWeights::F16(mut a), CpuWeights::F16(b)) => {
                    a.extend(b);
                    CpuWeights::F16(a)
                }
                (CpuWeights::Q8(mut a), CpuWeights::Q8(b)) => {
                    a.extend(b);
                    CpuWeights::Q8(a)
                }
                _ => unreachable!(),
            });
            return Ok(Linear {
                kind: Kind::Cpu(fused),
                bias,
                out_features: n,
                in_features: k,
            });
        }
        if all_dense {
            let ts: Vec<Tensor> = parts
                .iter()
                .map(|p| match &p.kind {
                    Kind::Dense(t) => t.clone(),
                    _ => unreachable!(),
                })
                .collect();
            if let Ok(t) = Tensor::cat(&ts, 0) {
                return Ok(Linear {
                    kind: Kind::Dense(t),
                    bias,
                    out_features: n,
                    in_features: k,
                });
            }
        }
        Err(parts)
    }

    /// Weight bytes resident for this layer.
    pub fn bytes(&self) -> usize {
        match &self.kind {
            Kind::Cpu(w) => w.bytes(),
            Kind::Dense(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Kind::Quant(QMatMul::QTensor(q)) => q.storage_size_in_bytes(),
            Kind::Quant(QMatMul::Tensor(t)) | Kind::Quant(QMatMul::TensorF16(t)) => {
                t.elem_count() * t.dtype().size_in_bytes()
            }
        }
    }

    pub fn describe(&self) -> &'static str {
        match &self.kind {
            Kind::Cpu(w) => w.label(),
            Kind::Dense(_) => "dense",
            Kind::Quant(_) => "quantized",
        }
    }

    /// `x`: [..., in] → [..., out]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();
        let mut out_dims = dims.clone();
        *out_dims.last_mut().unwrap() = self.out_features;
        let y = match &self.kind {
            Kind::Cpu(w) => {
                let xv: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let n = self.out_features;
                if m >= GEMM_ROWS {
                    // Prefill: widen weights in bands and use the blocked GEMM.
                    gemm_banded(&xv, m, k, w, n, x.device())?.reshape(out_dims)?
                } else {
                    let mut out = vec![0f32; m * n];
                    cpu_kernels::matmul(&xv, m, k, w, n, &mut out);
                    Tensor::from_vec(out, out_dims, x.device())?
                }
            }
            Kind::Dense(w) => {
                let x2 = x.reshape((m, k))?;
                x2.matmul(&w.t()?)?.reshape(out_dims)?
            }
            Kind::Quant(q) => {
                if x.device().is_cpu() && m >= GEMM_ROWS {
                    if let QMatMul::QTensor(qt) = q {
                        let wf = qt.dequantize(&candle_core::Device::Cpu)?;
                        let x2 = x.reshape((m, k))?.to_dtype(DType::F32)?;
                        return self.add_bias(x2.matmul(&wf.t()?)?.reshape(out_dims)?);
                    }
                }
                q.forward(x)?
            }
        };
        self.add_bias(y)
    }

    fn add_bias(&self, y: Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(b) => Ok(y.broadcast_add(&b.to_dtype(y.dtype())?)?),
            None => Ok(y),
        }
    }

    /// Gather rows (embedding lookup) as `[ids.len(), in]` in `act` dtype.
    pub fn rows(&self, ids: &[u32], device: &Device, act: DType) -> Result<Tensor> {
        let k = self.in_features;
        match &self.kind {
            Kind::Cpu(w) => {
                let mut out = vec![0f32; ids.len() * k];
                for (i, &id) in ids.iter().enumerate() {
                    let id = id as usize;
                    if id >= self.out_features {
                        bail!(
                            "token id {id} is outside the vocabulary ({})",
                            self.out_features
                        );
                    }
                    cpu_kernels::row_f32(w, id, k, &mut out[i * k..i * k + k]);
                }
                Ok(Tensor::from_vec(out, (ids.len(), k), device)?.to_dtype(act)?)
            }
            Kind::Dense(t) => {
                let idx = Tensor::new(ids, t.device())?;
                Ok(t.index_select(&idx, 0)?.to_dtype(act)?)
            }
            Kind::Quant(QMatMul::QTensor(q)) => {
                // Dequantize only the needed rows' blocks.
                let full = q.dequantize(device)?;
                let idx = Tensor::new(ids, device)?;
                Ok(full.index_select(&idx, 0)?.to_dtype(act)?)
            }
            Kind::Quant(QMatMul::Tensor(t)) | Kind::Quant(QMatMul::TensorF16(t)) => {
                let idx = Tensor::new(ids, t.device())?;
                Ok(t.index_select(&idx, 0)?.to_dtype(act)?)
            }
        }
    }
}

/// Large-batch CPU matmul: widen weight bands to f32 and use candle's GEMM.
fn gemm_banded(
    x: &[f32],
    m: usize,
    k: usize,
    w: &CpuWeights,
    n: usize,
    dev: &Device,
) -> Result<Tensor> {
    let xt = Tensor::from_slice(x, (m, k), dev)?;
    if let CpuWeights::F32(v) = w {
        let wt = Tensor::from_slice(v, (n, k), dev)?;
        return Ok(xt.matmul(&wt.t()?)?);
    }
    const BAND: usize = 2048;
    let mut parts = Vec::with_capacity(n.div_ceil(BAND));
    for s in (0..n).step_by(BAND) {
        let e = (s + BAND).min(n);
        let wf = cpu_kernels::rows_f32(w, s, e, k);
        let wt = Tensor::from_vec(wf, (e - s, k), dev)?;
        parts.push(xt.matmul(&wt.t()?)?);
    }
    Ok(Tensor::cat(&parts, 1)?)
}

pub type SharedLinear = Arc<Linear>;
