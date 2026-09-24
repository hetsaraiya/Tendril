//! Execution-level model configuration parsed from HuggingFace config.json.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tendril_core::model::Arch;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    Silu,
    GeluTanh,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    Linear(f64),
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max: f64,
    },
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub arch: Arch,
    pub model_type: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position: usize,
    pub rms_eps: f64,
    pub rope_theta: f64,
    /// Rope base for sliding-window layers (Gemma 3 uses 10k locally).
    pub rope_local_theta: f64,
    pub rope_scaling: RopeScaling,
    pub rope_local_scaling: RopeScaling,
    pub tie_embeddings: bool,
    pub activation: Activation,
    pub attention_bias: bool,
    /// Per-head RMSNorm on q and k (Qwen3, Gemma 3).
    pub qk_norm: bool,
    /// Gemma family: norms scale by (1 + w), embeddings scaled by sqrt(hidden).
    pub gemma_norm: bool,
    /// Gemma 2/3 extra post-attention / post-feedforward norms.
    pub sandwich_norm: bool,
    pub attn_softcap: Option<f64>,
    pub final_softcap: Option<f64>,
    /// Attention scale = 1/sqrt(query_pre_attn_scalar) (Gemma 2/3), else 1/sqrt(head_dim).
    pub query_pre_attn_scalar: Option<f64>,
    /// Per layer sliding window (None = global).
    pub layer_window: Vec<Option<usize>>,
    /// Fused qkv_proj / gate_up_proj (Phi-3).
    pub fused_qkv: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    /// Stored dtype name ("bfloat16"...).
    pub torch_dtype: String,
}

fn u(v: &Value, k: &str) -> Option<usize> {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
}
fn f(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}

fn ids(v: Option<&Value>) -> Vec<u32> {
    match v {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as u32))
            .collect(),
        _ => vec![],
    }
}

fn parse_rope(v: &Value, default_theta: f64) -> (f64, RopeScaling) {
    let theta = f(v, "rope_theta").unwrap_or(default_theta);
    let kind = v
        .get("rope_type")
        .or_else(|| v.get("type"))
        .and_then(|x| x.as_str())
        .unwrap_or("default");
    let scaling = match kind {
        "linear" => RopeScaling::Linear(f(v, "factor").unwrap_or(1.0)),
        "llama3" => RopeScaling::Llama3 {
            factor: f(v, "factor").unwrap_or(8.0),
            low_freq_factor: f(v, "low_freq_factor").unwrap_or(1.0),
            high_freq_factor: f(v, "high_freq_factor").unwrap_or(4.0),
            original_max: f(v, "original_max_position_embeddings").unwrap_or(8192.0),
        },
        "default" => RopeScaling::None,
        other => {
            tracing::warn!("rope type '{other}' not implemented; using unscaled RoPE (accurate for contexts within the original training length)");
            RopeScaling::None
        }
    };
    (theta, scaling)
}

impl ModelConfig {
    pub fn from_json(raw: &Value) -> Result<ModelConfig> {
        let top_type = raw.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
        let c = raw
            .get("text_config")
            .filter(|t| t.is_object())
            .unwrap_or(raw);
        let model_type = c
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or(top_type)
            .to_string();
        let arch = Arch::from_hf(&model_type);
        if arch == Arch::Other {
            bail!(
                "model type '{model_type}' is not supported yet (supported: llama, mistral, qwen2, qwen3, gemma, gemma2, gemma3, phi3)"
            );
        }
        if c.get("num_local_experts").is_some() || c.get("num_experts").is_some() {
            bail!("mixture-of-experts models are not supported yet");
        }
        let hidden = u(c, "hidden_size").context("hidden_size missing")?;
        let heads = u(c, "num_attention_heads").context("num_attention_heads missing")?;
        let layers = u(c, "num_hidden_layers").context("num_hidden_layers missing")?;
        let head_dim = u(c, "head_dim").unwrap_or(hidden / heads);
        let gemma = matches!(arch, Arch::Gemma | Arch::Gemma2 | Arch::Gemma3);
        let activation = match c
            .get("hidden_activation")
            .or_else(|| c.get("hidden_act"))
            .and_then(|v| v.as_str())
            .unwrap_or(if gemma { "gelu_pytorch_tanh" } else { "silu" })
        {
            "silu" | "swish" => Activation::Silu,
            "gelu" | "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" => Activation::GeluTanh,
            other => bail!("unsupported activation '{other}'"),
        };
        // RoPE: classic keys (rope_theta / rope_scaling) or transformers v5
        // `rope_parameters`, which may be keyed by layer type.
        let default_theta = f(c, "rope_theta").unwrap_or(10000.0);
        let (rope_theta, rope_scaling, local) =
            match c.get("rope_parameters").filter(|v| v.is_object()) {
                Some(rp)
                    if rp.get("full_attention").is_some()
                        || rp.get("sliding_attention").is_some() =>
                {
                    let g = rp.get("full_attention").unwrap_or(rp);
                    let (gt, gs) = parse_rope(g, default_theta);
                    let local = rp.get("sliding_attention").map(|l| parse_rope(l, 10000.0));
                    (gt, gs, local)
                }
                Some(rp) => {
                    let (t, s) = parse_rope(rp, default_theta);
                    (t, s, None)
                }
                None => {
                    let scaling = c
                        .get("rope_scaling")
                        .filter(|v| v.is_object())
                        .map(|rs| parse_rope(rs, default_theta).1)
                        .unwrap_or(RopeScaling::None);
                    (default_theta, scaling, None)
                }
            };
        let (rope_local_theta, rope_local_scaling) = match local {
            Some((t, s)) => (t, s),
            None => (
                f(c, "rope_local_base_freq").unwrap_or(10000.0),
                RopeScaling::None,
            ),
        };
        let window = u(c, "sliding_window");
        let mut layer_window = vec![None; layers];
        if let Some(types) = c.get("layer_types").and_then(|v| v.as_array()) {
            for (i, t) in types.iter().enumerate().take(layers) {
                if t.as_str() == Some("sliding_attention") {
                    layer_window[i] = window;
                }
            }
        } else if let Some(w) = window {
            match arch {
                Arch::Gemma2 => (0..layers)
                    .filter(|i| i % 2 == 0)
                    .for_each(|i| layer_window[i] = Some(w)),
                Arch::Gemma3 => {
                    let p = u(c, "sliding_window_pattern").unwrap_or(6).max(1);
                    (0..layers)
                        .filter(|i| (i + 1) % p != 0)
                        .for_each(|i| layer_window[i] = Some(w));
                }
                Arch::Mistral => layer_window.iter_mut().for_each(|x| *x = Some(w)),
                Arch::Qwen2 | Arch::Qwen3 => {
                    if c.get("use_sliding_window").and_then(|v| v.as_bool()) == Some(true) {
                        let max_window_layers = u(c, "max_window_layers").unwrap_or(layers);
                        (0..layers)
                            .filter(|&i| i >= max_window_layers)
                            .for_each(|i| layer_window[i] = Some(w));
                    }
                }
                _ => {}
            }
        }
        let tie = c
            .get("tie_word_embeddings")
            .or_else(|| raw.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .unwrap_or(gemma);
        let mut eos = ids(c.get("eos_token_id").or_else(|| raw.get("eos_token_id")));
        eos.dedup();
        Ok(ModelConfig {
            arch,
            model_type,
            num_layers: layers,
            hidden_size: hidden,
            intermediate_size: u(c, "intermediate_size").context("intermediate_size missing")?,
            num_heads: heads,
            num_kv_heads: u(c, "num_key_value_heads").unwrap_or(heads),
            head_dim,
            vocab_size: u(c, "vocab_size")
                .or_else(|| u(raw, "vocab_size"))
                .context("vocab_size missing")?,
            max_position: u(c, "max_position_embeddings").unwrap_or(4096),
            rms_eps: f(c, "rms_norm_eps")
                .or_else(|| f(c, "layer_norm_eps"))
                .unwrap_or(1e-6),
            rope_theta,
            rope_local_theta,
            rope_local_scaling,
            rope_scaling,
            tie_embeddings: tie,
            activation,
            attention_bias: arch == Arch::Qwen2
                || c.get("attention_bias").and_then(|v| v.as_bool()) == Some(true),
            qk_norm: matches!(arch, Arch::Qwen3 | Arch::Gemma3),
            gemma_norm: gemma,
            sandwich_norm: matches!(arch, Arch::Gemma2 | Arch::Gemma3),
            attn_softcap: f(c, "attn_logit_softcapping"),
            final_softcap: f(c, "final_logit_softcapping"),
            query_pre_attn_scalar: if matches!(arch, Arch::Gemma2 | Arch::Gemma3) {
                f(c, "query_pre_attn_scalar")
            } else {
                None
            },
            layer_window,
            fused_qkv: arch == Arch::Phi3,
            bos_token_id: c
                .get("bos_token_id")
                .or_else(|| raw.get("bos_token_id"))
                .and_then(|v| v.as_u64())
                .map(|x| x as u32),
            eos_token_ids: eos,
            torch_dtype: c
                .get("torch_dtype")
                .or_else(|| raw.get("torch_dtype"))
                .or_else(|| raw.get("dtype"))
                .and_then(|v| v.as_str())
                .unwrap_or("bfloat16")
                .to_string(),
        })
    }

    pub fn from_file(path: &std::path::Path) -> Result<ModelConfig> {
        let raw: Value = serde_json::from_slice(
            &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
        )
        .context("config.json is not valid JSON")?;
        Self::from_json(&raw)
    }

    pub fn attn_scale(&self) -> f64 {
        1.0 / self
            .query_pre_attn_scalar
            .unwrap_or(self.head_dim as f64)
            .sqrt()
    }
}
