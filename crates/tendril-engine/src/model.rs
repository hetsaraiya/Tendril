//! A pipeline stage: a contiguous slice of a decoder-only transformer.
//!
//! One implementation covers Llama, Mistral, Qwen2/3, Gemma 1/2/3 and Phi-3;
//! architecture differences are flags on [`ModelConfig`]. A stage owns only
//! its layers' weights and the KV cache of those layers, per sequence.

use crate::config::{Activation, ModelConfig, RopeScaling};
use crate::linear::{Linear, WeightFormat};
use crate::weights::WeightStore;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Which part of the model a stage runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageSpec {
    pub layer_start: usize,
    pub layer_end: usize,
    /// Owns the token embedding (first stage).
    pub embed: bool,
    /// Owns the final norm and output head (last stage).
    pub head: bool,
}

impl StageSpec {
    pub fn whole(cfg: &ModelConfig) -> StageSpec {
        StageSpec {
            layer_start: 0,
            layer_end: cfg.num_layers,
            embed: true,
            head: true,
        }
    }
    pub fn layers(&self) -> usize {
        self.layer_end - self.layer_start
    }
}

struct RmsNorm {
    w: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(
        ws: &WeightStore,
        name: &str,
        cfg: &ModelConfig,
        dev: &Device,
        dtype: DType,
    ) -> Result<RmsNorm> {
        let mut w = ws.tensor(name, &Device::Cpu, DType::F32)?;
        if cfg.gemma_norm {
            w = (w + 1.0)?;
        }
        Ok(RmsNorm {
            w: w.to_dtype(dtype)?.to_device(dev)?,
            eps: cfg.rms_eps,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::rms_norm(
            &x.contiguous()?,
            &self.w,
            self.eps as f32,
        )?)
    }
}

/// Rotary embedding tables, grown on demand.
struct Rope {
    inv_freq: Vec<f32>,
    cos: Tensor,
    sin: Tensor,
    len: usize,
    dtype: DType,
    device: Device,
}

impl Rope {
    fn new(
        cfg: &ModelConfig,
        theta: f64,
        scaling: &RopeScaling,
        dtype: DType,
        device: &Device,
    ) -> Result<Rope> {
        let d = cfg.head_dim;
        let mut inv: Vec<f32> = (0..d / 2)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / d as f64)) as f32)
            .collect();
        match scaling {
            RopeScaling::None => {}
            RopeScaling::Linear(f) => inv.iter_mut().for_each(|x| *x /= *f as f32),
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max,
            } => {
                let low_wl = original_max / low_freq_factor;
                let high_wl = original_max / high_freq_factor;
                for x in inv.iter_mut() {
                    let wl = 2.0 * std::f64::consts::PI / *x as f64;
                    let f = *x as f64;
                    *x = if wl < high_wl {
                        f
                    } else if wl > low_wl {
                        f / factor
                    } else {
                        let smooth = (original_max / wl - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * f / factor + smooth * f
                    } as f32;
                }
            }
        }
        let mut r = Rope {
            inv_freq: inv,
            cos: Tensor::zeros((1, d / 2), dtype, device)?,
            sin: Tensor::zeros((1, d / 2), dtype, device)?,
            len: 0,
            dtype,
            device: device.clone(),
        };
        r.ensure(2048)?;
        Ok(r)
    }

    fn ensure(&mut self, needed: usize) -> Result<()> {
        if needed <= self.len {
            return Ok(());
        }
        let len = needed.next_power_of_two().max(2048);
        let half = self.inv_freq.len();
        let mut freqs = Vec::with_capacity(len * half);
        for p in 0..len {
            for &f in &self.inv_freq {
                freqs.push(p as f32 * f);
            }
        }
        let t = Tensor::from_vec(freqs, (len, half), &Device::Cpu)?;
        self.cos = t.cos()?.to_dtype(self.dtype)?.to_device(&self.device)?;
        self.sin = t.sin()?.to_dtype(self.dtype)?.to_device(&self.device)?;
        self.len = len;
        Ok(())
    }

    /// x: [1, heads, seq, head_dim]
    fn apply(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let seq = x.dim(2)?;
        self.ensure(pos + seq)?;
        let cos = self.cos.narrow(0, pos, seq)?;
        let sin = self.sin.narrow(0, pos, seq)?;
        Ok(candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)?)
    }
}

/// Per-sequence KV for one layer. Sliding-window layers drop old entries.
struct LayerKv {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
    /// Absolute position of cached entry 0.
    offset: usize,
}

impl LayerKv {
    fn new() -> LayerKv {
        LayerKv {
            k: None,
            v: None,
            len: 0,
            offset: 0,
        }
    }

    /// Append [1, kv, seq, hd] and return views of all cached entries.
    fn append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let seq = k.dim(2)?;
        // Sliding-window layers: drop entries no query can see any more, before
        // appending (the returned views alias the cache storage).
        if let (Some(w), Some(kk), Some(vv)) = (window, &self.k, &self.v) {
            // Slack amortizes the copy; small windows compact often (and get tested).
            if self.len > w + (w / 4).clamp(1, 1024) {
                let drop = self.len - w;
                let keep_k = kk.narrow(2, drop, w)?.contiguous()?;
                let keep_v = vv.narrow(2, drop, w)?.contiguous()?;
                kk.slice_set(&keep_k, 2, 0)?;
                vv.slice_set(&keep_v, 2, 0)?;
                self.len = w;
                self.offset += drop;
            }
        }
        let cap = self.k.as_ref().map(|t| t.dim(2)).transpose()?.unwrap_or(0);
        if self.len + seq > cap {
            let new_cap = (self.len + seq).next_power_of_two().max(256);
            let (b, h, _, d) = k.dims4()?;
            let nk = Tensor::zeros((b, h, new_cap, d), k.dtype(), k.device())?;
            let nv = Tensor::zeros((b, h, new_cap, d), v.dtype(), v.device())?;
            if let (Some(ok), Some(ov)) = (&self.k, &self.v) {
                if self.len > 0 {
                    nk.slice_set(&ok.narrow(2, 0, self.len)?.contiguous()?, 2, 0)?;
                    nv.slice_set(&ov.narrow(2, 0, self.len)?.contiguous()?, 2, 0)?;
                }
            }
            self.k = Some(nk);
            self.v = Some(nv);
        }
        let kk = self.k.as_ref().unwrap();
        let vv = self.v.as_ref().unwrap();
        kk.slice_set(&k.contiguous()?, 2, self.len)?;
        vv.slice_set(&v.contiguous()?, 2, self.len)?;
        self.len += seq;
        Ok((kk.narrow(2, 0, self.len)?, vv.narrow(2, 0, self.len)?))
    }

    fn bytes(&self) -> usize {
        self.k
            .as_ref()
            .map(|t| 2 * t.elem_count() * t.dtype().size_in_bytes())
            .unwrap_or(0)
    }
}

/// Q, K and V projections: fused into one matmul when possible.
enum Qkv {
    Fused(Linear),
    Split(Linear, Linear, Linear),
}

/// Gate and up projections, fused when possible.
enum GateUp {
    Fused(Linear),
    Split(Linear, Linear),
}

struct Attention {
    qkv: Qkv,
    o: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
}

struct Mlp {
    gate_up: GateUp,
    down: Linear,
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ff_norm: Option<RmsNorm>,
    post_ff_norm: Option<RmsNorm>,
    window: Option<usize>,
    local_rope: bool,
}

/// What flows into a stage.
pub enum StageInput {
    Tokens(Vec<u32>),
    /// [1, seq, hidden]
    Hidden(Tensor),
}

/// What flows out of a stage.
pub enum StageOutput {
    Hidden(Tensor),
    /// f32 logits of the last position.
    Logits(Vec<f32>),
    /// f32 logits of every position (speculative verification).
    AllLogits(Vec<Vec<f32>>),
    /// Last stage asked not to compute logits (non-final prefill chunk).
    Nothing,
}

pub struct SeqState {
    kv: Vec<LayerKv>,
    /// Next absolute position.
    pub pos: usize,
}

pub struct LoadOptions {
    pub format: WeightFormat,
    pub device: Device,
    pub dtype: DType,
}

pub struct Stage {
    pub cfg: Arc<ModelConfig>,
    pub spec: StageSpec,
    pub device: Device,
    pub dtype: DType,
    embed: Option<Linear>,
    layers: Vec<Layer>,
    norm: Option<RmsNorm>,
    head: Option<Linear>,
    rope: Rope,
    rope_local: Option<Rope>,
    seqs: HashMap<u64, SeqState>,
    pub weight_bytes: usize,
}

fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

impl Stage {
    pub fn load(
        cfg: Arc<ModelConfig>,
        ws: &WeightStore,
        spec: StageSpec,
        opts: &LoadOptions,
    ) -> Result<Stage> {
        if spec.layer_end > cfg.num_layers || spec.layer_start > spec.layer_end {
            bail!(
                "stage layers {}..{} out of range (model has {})",
                spec.layer_start,
                spec.layer_end,
                cfg.num_layers
            );
        }
        let dev = &opts.device;
        let dt = opts.dtype;
        let fmt = opts.format;
        let n = |s: &str| ws.name(s);
        let mut weight_bytes = 0usize;

        // The embedding table is also the output head for tied models; load it
        // once and share when this stage needs both.
        let embed_name = n("embed_tokens.weight");
        let embed = if spec.embed {
            // Embeddings are a lookup table: keep native precision unless quantizing hard.
            let efmt = if matches!(fmt, WeightFormat::Native) {
                WeightFormat::Native
            } else {
                WeightFormat::Q8_0
            };
            let e =
                Linear::load(ws, &embed_name, None, dev, dt, efmt).context("loading embeddings")?;
            weight_bytes += e.bytes();
            Some(e)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(spec.layers());
        for i in spec.layer_start..spec.layer_end {
            let p = |s: &str| n(&format!("layers.{i}.{s}"));
            let lin = |name: &str, bias: bool| -> Result<Linear> {
                let full = p(name);
                let b = format!("{}.bias", full.trim_end_matches(".weight"));
                Linear::load(
                    ws,
                    &full,
                    if bias { Some(b.as_str()) } else { None },
                    dev,
                    dt,
                    fmt,
                )
            };
            let (q, k, v) = if cfg.fused_qkv {
                let w = ws.tensor(&p("self_attn.qkv_proj.weight"), &Device::Cpu, DType::F32)?;
                let qd = cfg.num_heads * cfg.head_dim;
                let kd = cfg.num_kv_heads * cfg.head_dim;
                let native = ws.raw(&p("self_attn.qkv_proj.weight"))?.0;
                (
                    Linear::from_f32(&w.narrow(0, 0, qd)?, None, dev, dt, fmt, native)?,
                    Linear::from_f32(&w.narrow(0, qd, kd)?, None, dev, dt, fmt, native)?,
                    Linear::from_f32(&w.narrow(0, qd + kd, kd)?, None, dev, dt, fmt, native)?,
                )
            } else {
                (
                    lin("self_attn.q_proj.weight", cfg.attention_bias)?,
                    lin("self_attn.k_proj.weight", cfg.attention_bias)?,
                    lin("self_attn.v_proj.weight", cfg.attention_bias)?,
                )
            };
            let o = lin("self_attn.o_proj.weight", false)?;
            let (gate, up) = if cfg.fused_qkv {
                let w = ws.tensor(&p("mlp.gate_up_proj.weight"), &Device::Cpu, DType::F32)?;
                let native = ws.raw(&p("mlp.gate_up_proj.weight"))?.0;
                let h = cfg.intermediate_size;
                (
                    Linear::from_f32(&w.narrow(0, 0, h)?, None, dev, dt, fmt, native)?,
                    Linear::from_f32(&w.narrow(0, h, h)?, None, dev, dt, fmt, native)?,
                )
            } else {
                (
                    lin("mlp.gate_proj.weight", false)?,
                    lin("mlp.up_proj.weight", false)?,
                )
            };
            let down = lin("mlp.down_proj.weight", false)?;
            let norm = |name: &str| RmsNorm::load(ws, &p(name), &cfg, dev, dt);
            let (q_norm, k_norm) = if cfg.qk_norm {
                (
                    Some(norm("self_attn.q_norm.weight")?),
                    Some(norm("self_attn.k_norm.weight")?),
                )
            } else {
                (None, None)
            };
            let (post_attn_norm, pre_ff_norm, post_ff_norm) = if cfg.sandwich_norm {
                (
                    norm("post_attention_layernorm.weight")?,
                    Some(norm("pre_feedforward_layernorm.weight")?),
                    Some(norm("post_feedforward_layernorm.weight")?),
                )
            } else {
                (norm("post_attention_layernorm.weight")?, None, None)
            };
            let qkv = match Linear::fuse(vec![q, k, v]) {
                Ok(f) => Qkv::Fused(f),
                Err(mut v) => {
                    let vv = v.pop().unwrap();
                    let kk = v.pop().unwrap();
                    let qq = v.pop().unwrap();
                    Qkv::Split(qq, kk, vv)
                }
            };
            let gate_up = match Linear::fuse(vec![gate, up]) {
                Ok(f) => GateUp::Fused(f),
                Err(mut v) => {
                    let u = v.pop().unwrap();
                    let g = v.pop().unwrap();
                    GateUp::Split(g, u)
                }
            };
            for l in [&o, &down] {
                weight_bytes += l.bytes();
            }
            weight_bytes += match &qkv {
                Qkv::Fused(f) => f.bytes(),
                Qkv::Split(a, b, c) => a.bytes() + b.bytes() + c.bytes(),
            };
            weight_bytes += match &gate_up {
                GateUp::Fused(f) => f.bytes(),
                GateUp::Split(a, b) => a.bytes() + b.bytes(),
            };
            let layer = Layer {
                attn: Attention {
                    qkv,
                    o,
                    q_norm,
                    k_norm,
                },
                mlp: Mlp { gate_up, down },
                input_norm: norm("input_layernorm.weight")?,
                post_attn_norm,
                pre_ff_norm,
                post_ff_norm,
                window: cfg.layer_window.get(i).copied().flatten(),
                local_rope: cfg.arch == tendril_core::model::Arch::Gemma3
                    && cfg.layer_window.get(i).copied().flatten().is_some(),
            };
            layers.push(layer);
        }

        let (norm, head) = if spec.head {
            let norm = RmsNorm::load(ws, &n("norm.weight"), &cfg, dev, dt)?;
            let head_name = match ws.head_name() {
                Some(h) if !cfg.tie_embeddings => h.to_string(),
                Some(h) if ws.has(h) && !ws.has(&embed_name) => h.to_string(),
                _ => embed_name.clone(),
            };
            let head =
                Linear::load(ws, &head_name, None, dev, dt, fmt).context("loading output head")?;
            weight_bytes += head.bytes();
            (Some(norm), Some(head))
        } else {
            (None, None)
        };

        let rope = Rope::new(&cfg, cfg.rope_theta, &cfg.rope_scaling, dt, dev)?;
        let rope_local = if cfg.arch == tendril_core::model::Arch::Gemma3 {
            Some(Rope::new(
                &cfg,
                cfg.rope_local_theta,
                &cfg.rope_local_scaling,
                dt,
                dev,
            )?)
        } else {
            None
        };
        Ok(Stage {
            cfg,
            spec,
            device: dev.clone(),
            dtype: dt,
            embed,
            layers,
            norm,
            head,
            rope,
            rope_local,
            seqs: HashMap::new(),
            weight_bytes,
        })
    }

    pub fn has_seq(&self, id: u64) -> bool {
        self.seqs.contains_key(&id)
    }

    pub fn seq_pos(&self, id: u64) -> Option<usize> {
        self.seqs.get(&id).map(|s| s.pos)
    }

    pub fn release(&mut self, id: u64) {
        self.seqs.remove(&id);
    }

    /// Keep only the first `len` positions of a sequence, so a new request that
    /// shares that prefix can continue from it without re-running prefill.
    pub fn truncate(&mut self, id: u64, len: usize) -> Result<()> {
        let st = self.seqs.get_mut(&id).context("unknown sequence")?;
        if len > st.pos {
            bail!(
                "cannot truncate sequence {id} to {len}: it only has {} positions",
                st.pos
            );
        }
        for kv in st.kv.iter() {
            if len < kv.offset {
                bail!("the start of this conversation already left the sliding-window cache");
            }
        }
        for kv in st.kv.iter_mut() {
            kv.len = len - kv.offset;
        }
        st.pos = len;
        Ok(())
    }

    /// Write a sequence's KV cache to `path` and free it from memory.
    pub fn spill(&mut self, id: u64, path: &std::path::Path) -> Result<u64> {
        let st = self.seqs.get(&id).context("unknown sequence")?;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"TKV1");
        buf.extend_from_slice(&(st.pos as u64).to_le_bytes());
        buf.extend_from_slice(&(st.kv.len() as u32).to_le_bytes());
        for kv in &st.kv {
            buf.extend_from_slice(&(kv.offset as u64).to_le_bytes());
            buf.extend_from_slice(&(kv.len as u64).to_le_bytes());
            match (&kv.k, &kv.v) {
                (Some(k), Some(v)) if kv.len > 0 => {
                    for t in [k, v] {
                        let part = t.narrow(2, 0, kv.len)?.contiguous()?;
                        let (code, bytes) = tensor_bytes(&part)?;
                        let dims = part.dims4()?;
                        buf.extend_from_slice(&code.to_le_bytes());
                        buf.extend_from_slice(&(dims.1 as u32).to_le_bytes());
                        buf.extend_from_slice(&(dims.3 as u32).to_le_bytes());
                        buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                        buf.extend_from_slice(&bytes);
                    }
                }
                _ => buf.extend_from_slice(&u32::MAX.to_le_bytes()),
            }
        }
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(path, &buf).with_context(|| format!("write {}", path.display()))?;
        self.seqs.remove(&id);
        Ok(buf.len() as u64)
    }

    /// Bring a spilled sequence back into memory.
    pub fn restore(&mut self, id: u64, path: &std::path::Path) -> Result<()> {
        let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let mut r = Cursor { b: &data, i: 0 };
        if r.take(4)? != b"TKV1" {
            bail!("not a Tendril KV file");
        }
        let pos = r.u64()? as usize;
        let layers = r.u32()? as usize;
        if layers != self.layers.len() {
            bail!(
                "KV file has {layers} layers, this stage has {}",
                self.layers.len()
            );
        }
        let mut kvs = Vec::with_capacity(layers);
        for _ in 0..layers {
            let offset = r.u64()? as usize;
            let len = r.u64()? as usize;
            let mut lk = LayerKv::new();
            lk.offset = offset;
            let code = r.u32()?;
            if code != u32::MAX {
                let mut parts = Vec::new();
                let mut c = code;
                for _ in 0..2 {
                    let heads = r.u32()? as usize;
                    let hd = r.u32()? as usize;
                    let n = r.u64()? as usize;
                    let bytes = r.take(n)?;
                    parts.push(
                        tensor_from_bytes(c, bytes, (1, heads, len, hd), &self.device)?
                            .to_dtype(self.dtype)?,
                    );
                    if parts.len() == 1 {
                        c = r.u32()?;
                    }
                }
                let cap = len.next_power_of_two().max(256);
                let (_, h, _, d) = parts[0].dims4()?;
                let k = Tensor::zeros((1, h, cap, d), self.dtype, &self.device)?;
                let v = Tensor::zeros((1, h, cap, d), self.dtype, &self.device)?;
                k.slice_set(&parts[0], 2, 0)?;
                v.slice_set(&parts[1], 2, 0)?;
                lk.k = Some(k);
                lk.v = Some(v);
                lk.len = len;
            }
            kvs.push(lk);
        }
        self.seqs.insert(id, SeqState { kv: kvs, pos });
        Ok(())
    }

    pub fn active_seqs(&self) -> usize {
        self.seqs.len()
    }

    pub fn kv_bytes(&self) -> usize {
        self.seqs
            .values()
            .flat_map(|s| s.kv.iter())
            .map(|k| k.bytes())
            .sum()
    }

    /// Run this stage for `seq` starting at absolute position `pos`.
    /// `want_logits` lets non-final prefill chunks skip the output head.
    pub fn forward(
        &mut self,
        seq: u64,
        pos: usize,
        input: StageInput,
        want_logits: bool,
    ) -> Result<StageOutput> {
        self.forward_batch(vec![BatchItem {
            seq,
            pos,
            input,
            want_logits,
            all_logits: false,
        }])
        .pop()
        .expect("one result per item")
    }

    /// Run several sequences through this stage in one pass: every weight is
    /// read once for the whole batch, while attention and KV stay per sequence.
    /// Items may be decode steps or prefill chunks; a sequence may appear once.
    pub fn forward_batch(&mut self, items: Vec<BatchItem>) -> Vec<Result<StageOutput>> {
        let n = items.len();
        let mut results: Vec<Option<Result<StageOutput>>> = (0..n).map(|_| None).collect();
        let n_layers = self.layers.len();
        // Validate and take each sequence's state out of the map.
        let mut taken: Vec<(usize, u64, SeqState)> = Vec::new();
        let mut inputs: Vec<(usize, StageInput, (bool, bool))> = Vec::new();
        for (i, it) in items.into_iter().enumerate() {
            if taken.iter().any(|(_, s, _)| *s == it.seq) {
                results[i] = Some(Err(anyhow::anyhow!(
                    "sequence {} appears twice in one batch",
                    it.seq
                )));
                continue;
            }
            let state = self.seqs.remove(&it.seq).unwrap_or_else(|| SeqState {
                kv: (0..n_layers).map(|_| LayerKv::new()).collect(),
                pos: 0,
            });
            if state.pos != it.pos {
                let expected = state.pos;
                if expected > 0 {
                    self.seqs.insert(it.seq, state);
                }
                results[i] = Some(Err(anyhow::anyhow!(
                    "sequence {}: expected position {expected}, got {} (out-of-order or duplicate message)",
                    it.seq,
                    it.pos
                )));
                continue;
            }
            taken.push((i, it.seq, state));
            inputs.push((i, it.input, (it.want_logits, it.all_logits)));
        }
        if !taken.is_empty() {
            match self.run_batch(&mut taken, inputs) {
                Ok(outs) => {
                    for (i, o) in outs {
                        results[i] = Some(Ok(o));
                    }
                    for (_, seq, st) in taken {
                        self.seqs.insert(seq, st);
                    }
                }
                Err(e) => {
                    // KV may be partially updated: these sequences are lost.
                    let msg = format!("{e:#}");
                    for (i, _, _) in taken {
                        results[i] = Some(Err(anyhow::anyhow!("{msg}")));
                    }
                }
            }
        }
        results
            .into_iter()
            .map(|r| r.expect("every item gets a result"))
            .collect()
    }

    fn run_batch(
        &mut self,
        taken: &mut [(usize, u64, SeqState)],
        inputs: Vec<(usize, StageInput, (bool, bool))>,
    ) -> Result<Vec<(usize, StageOutput)>> {
        let cfg = self.cfg.clone();
        // Concatenate every item's tokens (or hidden rows) along the sequence axis.
        let mut spans: Vec<(usize, usize, usize)> = Vec::with_capacity(inputs.len()); // (pos, start, len)
        let mut want = Vec::with_capacity(inputs.len());
        let mut start = 0;
        let first_is_tokens = matches!(inputs.first().map(|x| &x.1), Some(StageInput::Tokens(_)));
        let mut ids: Vec<u32> = Vec::new();
        let mut hidden: Vec<Tensor> = Vec::new();
        for ((_, _, st), (_, input, w)) in taken.iter().zip(inputs) {
            let len = match input {
                StageInput::Tokens(t) => {
                    if !first_is_tokens {
                        bail!("cannot mix token and hidden-state inputs in one batch");
                    }
                    let l = t.len();
                    ids.extend(t);
                    l
                }
                StageInput::Hidden(h) => {
                    if first_is_tokens {
                        bail!("cannot mix token and hidden-state inputs in one batch");
                    }
                    let h = h.to_device(&self.device)?.to_dtype(self.dtype)?;
                    let l = h.dim(1)?;
                    hidden.push(h);
                    l
                }
            };
            if len == 0 {
                bail!("empty input");
            }
            spans.push((st.pos, start, len));
            want.push(w);
            start += len;
        }
        let mut x = if first_is_tokens {
            let e = self
                .embed
                .as_ref()
                .context("this stage does not own the embeddings")?;
            let t = e.rows(&ids, &self.device, self.dtype)?.unsqueeze(0)?;
            if cfg.gemma_norm {
                let s = Tensor::new((cfg.hidden_size as f32).sqrt(), &self.device)?
                    .to_dtype(self.dtype)?;
                t.broadcast_mul(&s)?
            } else {
                t
            }
        } else if hidden.len() == 1 {
            hidden.pop().unwrap()
        } else {
            Tensor::cat(&hidden, 1)?
        };
        for (li, layer) in self.layers.iter().enumerate() {
            let rope = if layer.local_rope {
                self.rope_local.as_mut().unwrap()
            } else {
                &mut self.rope
            };
            let mut kvs: Vec<&mut LayerKv> =
                taken.iter_mut().map(|(_, _, st)| &mut st.kv[li]).collect();
            x = layer_forward(layer, &cfg, rope, &mut kvs, &x, &spans)?;
        }
        for ((_, _, st), (_, _, len)) in taken.iter_mut().zip(&spans) {
            st.pos += len;
        }
        let idx: Vec<usize> = taken.iter().map(|(i, _, _)| *i).collect();
        if !self.spec.head {
            let mut out = Vec::with_capacity(spans.len());
            for (k, &(_, s, len)) in spans.iter().enumerate() {
                out.push((idx[k], StageOutput::Hidden(x.narrow(1, s, len)?)));
            }
            return Ok(out);
        }
        // Output head only on the rows that need logits: the last row of each
        // item, or every row when verifying speculative drafts.
        let mut rows: Vec<u32> = Vec::new();
        for ((_, s, len), (w, all)) in spans.iter().zip(&want) {
            if *all {
                rows.extend((*s..s + len).map(|r| r as u32));
            } else if *w {
                rows.push((s + len - 1) as u32);
            }
        }
        let mut logits_rows: Vec<Vec<f32>> = Vec::new();
        if !rows.is_empty() {
            let sel = Tensor::new(rows.as_slice(), &self.device)?;
            let last = x.index_select(&sel, 1)?;
            let h = self.norm.as_ref().unwrap().forward(&last)?;
            let mut logits = self
                .head
                .as_ref()
                .unwrap()
                .forward(&h)?
                .to_dtype(DType::F32)?;
            if let Some(cap) = cfg.final_softcap {
                logits = ((logits / cap)?.tanh()? * cap)?;
            }
            logits_rows = logits.squeeze(0)?.to_vec2()?;
        }
        let mut lr = logits_rows.into_iter();
        Ok(idx
            .into_iter()
            .zip(want)
            .zip(&spans)
            .map(|((i, (w, all)), (_, _, len))| {
                let out = if all {
                    StageOutput::AllLogits(
                        (0..*len).map(|_| lr.next().unwrap_or_default()).collect(),
                    )
                } else if w {
                    StageOutput::Logits(lr.next().unwrap_or_default())
                } else {
                    StageOutput::Nothing
                };
                (i, out)
            })
            .collect())
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.i + n > self.b.len() {
            bail!("truncated KV file");
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn tensor_bytes(t: &Tensor) -> Result<(u32, Vec<u8>)> {
    let flat = t.flatten_all()?;
    Ok(match t.dtype() {
        DType::F32 => (
            0,
            flat.to_vec1::<f32>()?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
        ),
        DType::BF16 => (
            1,
            flat.to_vec1::<half::bf16>()?
                .iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect(),
        ),
        DType::F16 => (
            2,
            flat.to_vec1::<half::f16>()?
                .iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect(),
        ),
        other => bail!("cannot store {other:?} KV"),
    })
}

fn tensor_from_bytes(
    code: u32,
    b: &[u8],
    shape: (usize, usize, usize, usize),
    dev: &Device,
) -> Result<Tensor> {
    let n = shape.0 * shape.1 * shape.2 * shape.3;
    Ok(match code {
        0 if b.len() == n * 4 => Tensor::from_vec(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>(),
            shape,
            dev,
        )?,
        1 if b.len() == n * 2 => Tensor::from_vec(
            b.chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect::<Vec<_>>(),
            shape,
            dev,
        )?,
        2 if b.len() == n * 2 => Tensor::from_vec(
            b.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect::<Vec<_>>(),
            shape,
            dev,
        )?,
        _ => bail!("corrupt KV file"),
    })
}

/// One sequence's work in a batched stage call.
pub struct BatchItem {
    pub seq: u64,
    pub pos: usize,
    pub input: StageInput,
    pub want_logits: bool,
    /// Logits for every input position, not just the last.
    pub all_logits: bool,
}

fn layer_forward(
    l: &Layer,
    cfg: &ModelConfig,
    rope: &mut Rope,
    kvs: &mut [&mut LayerKv],
    x: &Tensor,
    spans: &[(usize, usize, usize)],
) -> Result<Tensor> {
    let residual = x;
    let h = l.input_norm.forward(x)?;
    let a = attention(l, cfg, rope, kvs, &h, spans)?;
    let x = if cfg.sandwich_norm {
        (residual + l.post_attn_norm.forward(&a)?)?
    } else {
        (residual + a)?
    };
    let h = match &l.pre_ff_norm {
        Some(n) => n.forward(&x)?,
        None => l.post_attn_norm.forward(&x)?,
    };
    let (gate, up) = match &l.mlp.gate_up {
        GateUp::Fused(f) => {
            let gu = f.forward(&h)?;
            let i = cfg.intermediate_size;
            (gu.narrow(2, 0, i)?, gu.narrow(2, i, i)?)
        }
        GateUp::Split(g, u) => (g.forward(&h)?, u.forward(&h)?),
    };
    let act = match cfg.activation {
        Activation::Silu => candle_nn::ops::silu(&gate)?,
        Activation::GeluTanh => gelu_tanh(&gate)?,
    };
    let m = l.mlp.down.forward(&(act * up)?)?;
    let m = match &l.post_ff_norm {
        Some(n) => n.forward(&m)?,
        None => m,
    };
    Ok((x + m)?)
}

fn attention(
    l: &Layer,
    cfg: &ModelConfig,
    rope: &mut Rope,
    kvs: &mut [&mut LayerKv],
    x: &Tensor,
    spans: &[(usize, usize, usize)],
) -> Result<Tensor> {
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    // Projections for every token of every sequence in one matmul.
    let (q_all, k_all_new, v_all_new) = match &l.attn.qkv {
        Qkv::Fused(f) => {
            let y = f.forward(x)?;
            (
                y.narrow(2, 0, nh * hd)?,
                y.narrow(2, nh * hd, nkv * hd)?,
                y.narrow(2, (nh + nkv) * hd, nkv * hd)?,
            )
        }
        Qkv::Split(q, k, v) => (q.forward(x)?, k.forward(x)?, v.forward(x)?),
    };
    let mut outs = Vec::with_capacity(spans.len());
    for (kv, &(pos, start, seq)) in kvs.iter_mut().zip(spans) {
        let q = q_all.narrow(1, start, seq)?;
        let k = k_all_new.narrow(1, start, seq)?;
        let v = v_all_new.narrow(1, start, seq)?;
        outs.push(attend_one(l, cfg, rope, kv, &q, &k, &v, pos)?);
    }
    let out = if outs.len() == 1 {
        outs.pop().unwrap()
    } else {
        Tensor::cat(&outs, 1)?
    };
    l.attn.o.forward(&out)
}

/// Attention for one sequence's rows: q/k/v are [1, seq, heads*hd] projections.
/// Returns [1, seq, heads*hd] before the output projection.
fn attend_one(
    l: &Layer,
    cfg: &ModelConfig,
    rope: &mut Rope,
    kv: &mut LayerKv,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pos: usize,
) -> Result<Tensor> {
    let (b, seq, _) = q.dims3()?;
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let x = q;
    let q = q.reshape((b, seq, nh, hd))?.transpose(1, 2)?;
    let k = k.reshape((b, seq, nkv, hd))?.transpose(1, 2)?;
    let v = v.reshape((b, seq, nkv, hd))?.transpose(1, 2)?;
    let (q, k) = match (&l.attn.q_norm, &l.attn.k_norm) {
        (Some(qn), Some(kn)) => (qn.forward(&q)?, kn.forward(&k)?),
        _ => (q, k),
    };
    let q = rope.apply(&q, pos)?;
    let k = rope.apply(&k, pos)?;
    let (k_all, v_all) = kv.append(&k, &v, l.window)?;
    let total = k_all.dim(2)?;
    // Cached keys are contiguous positions ending at the current token.
    let first_key_pos = kv.offset;
    debug_assert_eq!(first_key_pos + total, pos + seq);

    // CPU: stream attention straight out of the cache buffers.
    if x.device().is_cpu() && q.dtype() == DType::F32 {
        if let (Some(kbuf), Some(vbuf)) = (kv.k.as_ref(), kv.v.as_ref()) {
            let params = crate::attention::AttnParams {
                heads: nh,
                kv_heads: nkv,
                head_dim: hd,
                seq,
                pos,
                key_offset: first_key_pos,
                len: kv.len,
                cap: kbuf.dim(2)?,
                scale: cfg.attn_scale() as f32,
                softcap: cfg.attn_softcap.map(|c| c as f32),
                window: l.window,
            };
            let out = crate::attention::attend(&q.contiguous()?, kbuf, vbuf, &params)?;
            let out = Tensor::from_vec(out, (b, seq, nh * hd), x.device())?;
            return Ok(out);
        }
    }

    // Grouped-query attention without copying K/V: fold query groups into rows.
    let groups = nh / nkv;
    let q = (q.contiguous()? * cfg.attn_scale())?;
    let q = q.reshape((b, nkv, groups * seq, hd))?;
    let mut scores = q
        .matmul(&k_all.transpose(2, 3)?.contiguous()?)?
        .to_dtype(DType::F32)?;
    if let Some(cap) = cfg.attn_softcap {
        scores = ((scores / cap)?.tanh()? * cap)?;
    }
    let needs_mask = seq > 1 || l.window.is_some_and(|w| total > w);
    if needs_mask {
        let mut m = vec![0f32; seq * total];
        for i in 0..seq {
            let qp = pos + i;
            for j in 0..total {
                let kp = first_key_pos + j;
                let visible = kp <= qp && l.window.is_none_or(|w| qp - kp < w);
                if !visible {
                    m[i * total + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(m, (seq, total), x.device())?;
        // Rows are ordered (group, seq): repeat the mask per group.
        let mask = mask
            .unsqueeze(0)?
            .repeat((groups, 1, 1))?
            .reshape((groups * seq, total))?;
        scores = scores.broadcast_add(&mask)?;
    }
    let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v_all.dtype())?;
    let out = probs.matmul(&v_all.contiguous()?)?; // [b, nkv, groups*seq, hd]
    let out = out
        .reshape((b, nh, seq, hd))?
        .transpose(1, 2)?
        .reshape((b, seq, nh * hd))?;
    Ok(out)
}
