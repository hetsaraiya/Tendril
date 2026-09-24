//! Parse a HuggingFace `config.json` into a [`ModelSpec`] using parameter
//! formulas. When real tensor headers are available they replace the
//! formula-derived byte counts (see `source.rs`).

use super::{Arch, Attention, ComponentBytes, ModelSpec, MoeInfo, ParamCounts, Quant};
use anyhow::{bail, Context, Result};
use serde_json::Value;

fn get_u64(v: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| v.get(*k).and_then(|x| x.as_u64()))
}

/// Build a spec from config.json content.
pub fn spec_from_config(id: &str, origin: &str, raw: &Value) -> Result<ModelSpec> {
    // Multimodal wrappers (Gemma 3, Mistral 3...) nest the LM in text_config.
    let top_type = raw.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let cfg = raw
        .get("text_config")
        .filter(|t| t.is_object())
        .unwrap_or(raw);
    let model_type = cfg
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or(top_type)
        .to_string();
    let arch = Arch::from_hf(&model_type);

    let num_layers = get_u64(cfg, &["num_hidden_layers", "n_layer", "num_layers"])
        .context("config.json has no num_hidden_layers")?;
    let hidden = get_u64(cfg, &["hidden_size", "n_embd", "d_model"])
        .context("config.json has no hidden_size")?;
    let num_heads = get_u64(cfg, &["num_attention_heads", "n_head"])
        .context("config.json has no num_attention_heads")?;
    let num_kv = get_u64(cfg, &["num_key_value_heads"]).unwrap_or(num_heads);
    let head_dim = get_u64(cfg, &["head_dim"]).unwrap_or(hidden / num_heads.max(1));
    let inter = get_u64(cfg, &["intermediate_size", "n_inner"]).unwrap_or(4 * hidden);
    let vocab = get_u64(cfg, &["vocab_size"])
        .or_else(|| get_u64(raw, &["vocab_size"]))
        .unwrap_or(32000);
    let max_pos = get_u64(cfg, &["max_position_embeddings", "n_positions"]).unwrap_or(4096);
    if num_layers == 0 || hidden == 0 || num_heads == 0 {
        bail!("config.json has zero-sized dimensions");
    }
    let default_tie = matches!(arch, Arch::Gemma | Arch::Gemma2 | Arch::Gemma3);
    let tie = cfg
        .get("tie_word_embeddings")
        .or_else(|| raw.get("tie_word_embeddings"))
        .and_then(|v| v.as_bool())
        .unwrap_or(default_tie);

    // Attention span per layer.
    let window = cfg.get("sliding_window").and_then(|v| v.as_u64());
    let mut attention = vec![Attention::Global; num_layers as usize];
    if let Some(types) = cfg.get("layer_types").and_then(|v| v.as_array()) {
        for (i, t) in types.iter().enumerate().take(num_layers as usize) {
            if t.as_str() == Some("sliding_attention") {
                if let Some(w) = window {
                    attention[i] = Attention::Sliding(w);
                }
            }
        }
    } else if let Some(w) = window {
        match arch {
            Arch::Gemma2 => {
                for (i, a) in attention.iter_mut().enumerate() {
                    if i % 2 == 0 {
                        *a = Attention::Sliding(w);
                    }
                }
            }
            Arch::Gemma3 => {
                let pattern = get_u64(cfg, &["sliding_window_pattern"])
                    .unwrap_or(6)
                    .max(1);
                for (i, a) in attention.iter_mut().enumerate() {
                    if !(i as u64 + 1).is_multiple_of(pattern) {
                        *a = Attention::Sliding(w);
                    }
                }
            }
            Arch::Mistral => attention
                .iter_mut()
                .for_each(|a| *a = Attention::Sliding(w)),
            Arch::Qwen2 | Arch::Qwen3 => {
                if cfg.get("use_sliding_window").and_then(|v| v.as_bool()) == Some(true) {
                    attention
                        .iter_mut()
                        .for_each(|a| *a = Attention::Sliding(w));
                }
            }
            _ => {}
        }
    }

    // Mixture of experts.
    let experts = get_u64(
        cfg,
        &["num_local_experts", "num_experts", "n_routed_experts"],
    )
    .filter(|&e| e > 1);
    let moe = experts.map(|e| MoeInfo {
        experts: e,
        active: get_u64(cfg, &["num_experts_per_tok", "moe_topk"]).unwrap_or(2),
    });
    let moe_inter = get_u64(cfg, &["moe_intermediate_size"]).unwrap_or(inter);

    // Parameter formulas.
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv * head_dim;
    let mut attn = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden;
    let mut small = 0u64;
    if arch == Arch::Qwen2 || cfg.get("attention_bias").and_then(|v| v.as_bool()) == Some(true) {
        attn += q_dim + 2 * kv_dim;
        small += q_dim + 2 * kv_dim;
    }
    let mlp = match &moe {
        Some(m) => m.experts * 3 * hidden * moe_inter + hidden * m.experts,
        None => 3 * hidden * inter,
    };
    let norms = match arch {
        Arch::Gemma2 => 4 * hidden,
        Arch::Gemma3 => 4 * hidden + 2 * head_dim,
        Arch::Qwen3 => 2 * hidden + 2 * head_dim,
        _ => 2 * hidden,
    };
    small += norms;
    let per_layer = attn + mlp + norms;
    let params = ParamCounts {
        embed: vocab * hidden,
        layers: vec![per_layer; num_layers as usize],
        layer_small: vec![small; num_layers as usize],
        final_norm: hidden,
        lm_head: if tie { 0 } else { vocab * hidden },
    };

    let stored = stored_quant(raw);
    let mut notes = Vec::new();
    if moe.is_some() {
        notes.push("Mixture-of-experts: memory is planned for all experts; Tendril cannot execute MoE models yet.".into());
    }
    if !arch.executable() {
        notes.push(format!(
            "Architecture '{model_type}' is not supported by Tendril's engine yet; planning uses generic transformer formulas."
        ));
    }
    let mut spec = ModelSpec {
        id: id.to_string(),
        origin: origin.to_string(),
        arch,
        model_type,
        num_layers,
        hidden_size: hidden,
        intermediate_size: inter,
        num_heads,
        num_kv_heads: num_kv,
        head_dim,
        vocab_size: vocab,
        max_position: max_pos,
        tie_embeddings: tie,
        attention,
        params,
        stored,
        // Force a byte recompute below by starting from a sentinel repr.
        repr: Quant::Mixed,
        bytes: ComponentBytes::default(),
        bytes_measured: false,
        moe,
        notes,
    };
    spec = spec.with_repr(stored);
    spec.repr = stored;
    Ok(spec)
}

fn stored_quant(raw: &Value) -> Quant {
    // MLX-community quantized checkpoints.
    if let Some(q) = raw
        .get("quantization")
        .or_else(|| raw.get("quantization_config"))
    {
        if let Some(bits) = q.get("bits").and_then(|b| b.as_u64()) {
            return match bits {
                8 => Quant::Q8_0,
                6 => Quant::Q6K,
                5 => Quant::Q5K,
                _ => Quant::Q4K,
            };
        }
        if q.get("load_in_4bit").and_then(|v| v.as_bool()) == Some(true) {
            return Quant::Q4K;
        }
    }
    let dt = raw
        .get("torch_dtype")
        .or_else(|| raw.get("dtype"))
        .or_else(|| raw.get("text_config").and_then(|t| t.get("torch_dtype")))
        .and_then(|v| v.as_str())
        .unwrap_or("bfloat16");
    match dt {
        "float32" => Quant::F32,
        "float16" => Quant::F16,
        _ => Quant::Bf16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn llama_8b_params() {
        let cfg = json!({
            "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
            "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8,
            "vocab_size": 128256, "max_position_embeddings": 131072, "tie_word_embeddings": false,
            "torch_dtype": "bfloat16"
        });
        let s = spec_from_config("llama", "test", &cfg).unwrap();
        let p = s.total_params() as f64 / 1e9;
        assert!((p - 8.03).abs() < 0.05, "params {p}");
        // bf16 ≈ 14.96 GiB
        assert!(
            (s.weight_bytes().as_gib() - 14.96).abs() < 0.1,
            "{}",
            s.weight_bytes()
        );
        // KV per token per layer: 2 * 8 * 128 * 2 bytes = 4 KiB
        assert_eq!(s.kv_bytes_per_token_layer(2), 4096);
    }

    #[test]
    fn gemma2_sliding() {
        let cfg = json!({
            "model_type": "gemma2", "num_hidden_layers": 42, "hidden_size": 3584,
            "intermediate_size": 14336, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 256, "vocab_size": 256000, "sliding_window": 4096
        });
        let s = spec_from_config("g", "t", &cfg).unwrap();
        assert!(s.tie_embeddings);
        assert_eq!(s.attention[0], Attention::Sliding(4096));
        assert_eq!(s.attention[1], Attention::Global);
        let p = s.total_params() as f64 / 1e9;
        assert!((p - 9.24).abs() < 0.1, "params {p}");
        // Sliding layers cap KV at the window.
        let full = s.kv_bytes(0..42, 8192, 1, 2);
        let expected = s.kv_bytes_per_token_layer(2) * (21 * 8192 + 21 * 4096);
        assert_eq!(full.0, expected);
    }

    #[test]
    fn quantized_repr_shrinks() {
        let cfg = json!({"model_type":"qwen2","num_hidden_layers":28,"hidden_size":3584,
            "intermediate_size":18944,"num_attention_heads":28,"num_key_value_heads":4,
            "vocab_size":152064,"tie_word_embeddings":false});
        let s = spec_from_config("q", "t", &cfg).unwrap();
        let q4 = s.with_repr(Quant::Q4K);
        let ratio = q4.weight_bytes().0 as f64 / s.weight_bytes().0 as f64;
        assert!(ratio > 0.27 && ratio < 0.33, "{ratio}");
    }
}
