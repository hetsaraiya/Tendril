//! Model inspection: architecture, tensor sizes and memory requirements,
//! derived without loading weights.

pub mod catalog;
pub mod config;
pub mod gguf;
pub mod safetensors;
pub mod source;

use crate::units::Bytes;
use serde::{Deserialize, Serialize};

/// Weight representation. Planning with a representation other than the
/// stored one is only ever done when the user asks for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quant {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q6K,
    Q5K,
    Q4K,
    Q4_0,
    /// Mixed or unknown GGUF layouts; bytes come from the file.
    Mixed,
}

impl Quant {
    pub fn bits_per_weight(self) -> f64 {
        match self {
            Quant::F32 => 32.0,
            Quant::F16 | Quant::Bf16 => 16.0,
            Quant::Q8_0 => 8.5,
            Quant::Q6K => 6.5625,
            Quant::Q5K => 5.5,
            Quant::Q4K => 4.5,
            Quant::Q4_0 => 4.5,
            Quant::Mixed => 5.0,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Quant::F32 => "f32",
            Quant::F16 => "f16",
            Quant::Bf16 => "bf16",
            Quant::Q8_0 => "q8_0",
            Quant::Q6K => "q6_k",
            Quant::Q5K => "q5_k",
            Quant::Q4K => "q4_k",
            Quant::Q4_0 => "q4_0",
            Quant::Mixed => "mixed",
        }
    }
    pub fn parse(s: &str) -> Option<Quant> {
        Some(match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "f32" | "fp32" => Quant::F32,
            "f16" | "fp16" => Quant::F16,
            "bf16" => Quant::Bf16,
            "q8" | "q8_0" | "int8" | "8bit" => Quant::Q8_0,
            "q6" | "q6_k" => Quant::Q6K,
            "q5" | "q5_k" | "q5_k_m" => Quant::Q5K,
            "q4" | "q4_k" | "q4_k_m" | "int4" | "4bit" => Quant::Q4K,
            "q4_0" => Quant::Q4_0,
            _ => return None,
        })
    }
    pub fn is_quantized(self) -> bool {
        !matches!(self, Quant::F32 | Quant::F16 | Quant::Bf16)
    }
    /// A short, honest statement of the expected quality impact.
    pub fn quality_note(self) -> &'static str {
        match self {
            Quant::F32 | Quant::F16 | Quant::Bf16 => "full quality",
            Quant::Q8_0 => "near-lossless (typically <0.1 perplexity change)",
            Quant::Q6K => "very small quality loss",
            Quant::Q5K => "small quality loss",
            Quant::Q4K | Quant::Q4_0 => "noticeable but usually acceptable quality loss",
            Quant::Mixed => "as stored",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Arch {
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
    Gemma,
    Gemma2,
    Gemma3,
    Phi3,
    Other,
}

impl Arch {
    pub fn from_hf(model_type: &str) -> Arch {
        match model_type {
            "llama" => Arch::Llama,
            "mistral" => Arch::Mistral,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "gemma" => Arch::Gemma,
            "gemma2" => Arch::Gemma2,
            "gemma3" | "gemma3_text" => Arch::Gemma3,
            "phi3" => Arch::Phi3,
            _ => Arch::Other,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Arch::Llama => "Llama",
            Arch::Mistral => "Mistral",
            Arch::Qwen2 => "Qwen2",
            Arch::Qwen3 => "Qwen3",
            Arch::Gemma => "Gemma",
            Arch::Gemma2 => "Gemma 2",
            Arch::Gemma3 => "Gemma 3",
            Arch::Phi3 => "Phi-3",
            Arch::Other => "Unknown",
        }
    }
    /// Architectures Tendril's execution engine can run.
    pub fn executable(self) -> bool {
        !matches!(self, Arch::Other)
    }
}

/// Attention span of one layer, for KV sizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind", content = "window")]
pub enum Attention {
    Global,
    Sliding(u64),
}

/// Parameter counts per component. Bytes are derived from these and the
/// representation, unless measured from files.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ParamCounts {
    pub embed: u64,
    pub layers: Vec<u64>,
    /// Params in 1-D tensors (norms, biases) per layer; these stay high precision when quantizing.
    pub layer_small: Vec<u64>,
    pub final_norm: u64,
    /// Zero when the output head is tied to the embedding.
    pub lm_head: u64,
}

impl ParamCounts {
    pub fn total(&self) -> u64 {
        self.embed + self.layers.iter().sum::<u64>() + self.final_norm + self.lm_head
    }
}

/// Byte sizes per component in a concrete representation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ComponentBytes {
    pub embed: Bytes,
    pub layers: Vec<Bytes>,
    pub final_norm: Bytes,
    /// Separate output head bytes. When tied this is zero but the last stage
    /// still needs a copy of the embedding (see `head_bytes`).
    pub lm_head: Bytes,
    /// Largest single tensor (load-time transient).
    pub largest_tensor: Bytes,
}

impl ComponentBytes {
    pub fn total(&self) -> Bytes {
        self.embed + self.layers.iter().copied().sum::<Bytes>() + self.final_norm + self.lm_head
    }
}

/// Mixture-of-experts facts. Tendril plans memory for MoE models but cannot
/// execute them yet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoeInfo {
    pub experts: u64,
    pub active: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    /// How the user referred to the model.
    pub id: String,
    /// Where the facts came from ("catalog", "huggingface", "local safetensors", "gguf").
    pub origin: String,
    pub arch: Arch,
    pub model_type: String,
    pub num_layers: u64,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_heads: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub vocab_size: u64,
    pub max_position: u64,
    pub tie_embeddings: bool,
    pub attention: Vec<Attention>,
    pub params: ParamCounts,
    /// Representation the weights are stored in.
    pub stored: Quant,
    /// Representation being planned (equals `stored` unless the user asked).
    pub repr: Quant,
    pub bytes: ComponentBytes,
    /// True when `bytes` came from real tensor headers rather than formulas.
    pub bytes_measured: bool,
    #[serde(default)]
    pub moe: Option<MoeInfo>,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl ModelSpec {
    pub fn total_params(&self) -> u64 {
        self.params.total()
    }

    /// KV cache bytes for one token in one layer at the given KV element size.
    pub fn kv_bytes_per_token_layer(&self, kv_elem_bytes: u64) -> u64 {
        2 * self.num_kv_heads * self.head_dim * kv_elem_bytes
    }

    /// KV bytes for `layers` at `context` tokens and `seqs` concurrent sequences.
    pub fn kv_bytes(
        &self,
        layers: std::ops::Range<usize>,
        context: u64,
        seqs: u64,
        kv_elem: u64,
    ) -> Bytes {
        let per = self.kv_bytes_per_token_layer(kv_elem);
        let mut total: u64 = 0;
        for l in layers {
            let span = match self.attention.get(l).copied().unwrap_or(Attention::Global) {
                Attention::Global => context,
                Attention::Sliding(w) => context.min(w),
            };
            total = total.saturating_add(per.saturating_mul(span).saturating_mul(seqs));
        }
        Bytes(total)
    }

    /// Bytes the final stage needs for the output head.
    pub fn head_bytes(&self) -> Bytes {
        if self.tie_embeddings {
            self.bytes.embed
        } else {
            self.bytes.lm_head
        }
    }

    /// Re-derive byte sizes for a different representation.
    pub fn with_repr(&self, q: Quant) -> ModelSpec {
        let mut m = self.clone();
        if q == self.repr {
            return m;
        }
        m.repr = q;
        m.bytes_measured = false;
        let bpw = q.bits_per_weight();
        let conv = |p: u64| Bytes((p as f64 * bpw / 8.0).ceil() as u64);
        // Embeddings are kept at 8 bits or better when quantizing (they are a
        // lookup table and cheap to keep precise), mirroring common practice.
        let emb_q = if q.is_quantized() && bpw < 8.5 {
            Quant::Q8_0
        } else {
            q
        };
        let conv_emb = |p: u64| Bytes((p as f64 * emb_q.bits_per_weight() / 8.0).ceil() as u64);
        let small = |p: u64| Bytes(p * 4);
        m.bytes.embed = conv_emb(m.params.embed);
        m.bytes.layers = m
            .params
            .layers
            .iter()
            .zip(m.params.layer_small.iter().chain(std::iter::repeat(&0)))
            .map(|(&p, &s)| {
                conv(p.saturating_sub(s)) + if q.is_quantized() { small(s) } else { conv(s) }
            })
            .collect();
        m.bytes.final_norm = if q.is_quantized() {
            small(m.params.final_norm)
        } else {
            conv(m.params.final_norm)
        };
        m.bytes.lm_head = conv(m.params.lm_head);
        let biggest_layer_tensor = m.intermediate_size * m.hidden_size;
        m.bytes.largest_tensor = conv_emb(m.params.embed)
            .max(conv(biggest_layer_tensor))
            .max(m.bytes.lm_head);
        m
    }

    /// Weight bytes of the whole model in the planned representation.
    pub fn weight_bytes(&self) -> Bytes {
        self.bytes.total()
    }

    /// Activation bytes crossing a pipeline boundary per token (f16/bf16 wire).
    pub fn boundary_bytes_per_token(&self) -> u64 {
        self.hidden_size * 2
    }
}
