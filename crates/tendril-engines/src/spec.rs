//! Memory models for planning models the planner can't see inside.

use tendril_core::model::{Arch, ComponentBytes, ModelSpec, ParamCounts, Quant};
use tendril_core::Bytes;

/// A model that runs whole on one machine (image and audio models, registry
/// models): one indivisible block of `bytes`, plus `working` memory the
/// engine needs on top (activations, VAE decode, KV it manages itself).
pub fn opaque(id: &str, origin: &str, bytes: u64, working: u64) -> ModelSpec {
    ModelSpec {
        id: id.into(),
        origin: origin.into(),
        arch: Arch::Other,
        model_type: "opaque".into(),
        num_layers: 1,
        hidden_size: 1,
        intermediate_size: 1,
        num_heads: 1,
        num_kv_heads: 0,
        head_dim: 0,
        vocab_size: 0,
        max_position: 0,
        tie_embeddings: true,
        attention: vec![tendril_core::model::Attention::Global],
        params: ParamCounts {
            embed: 0,
            layers: vec![bytes / 2],
            layer_small: vec![0],
            final_norm: 0,
            lm_head: 0,
        },
        stored: Quant::Mixed,
        repr: Quant::Mixed,
        bytes: ComponentBytes {
            embed: Bytes(0),
            layers: vec![Bytes(bytes + working)],
            final_norm: Bytes(0),
            lm_head: Bytes(0),
            largest_tensor: Bytes(0),
        },
        bytes_measured: true,
        moe: None,
        notes: vec![],
    }
}

/// Roughly what a model of `bytes` on disk needs beyond its weights
/// (activations, buffers): 10% plus 512 MiB.
pub fn working_memory(bytes: u64) -> u64 {
    bytes / 10 + (512 << 20)
}

/// GGUF file names: split parts (`-00001-of-00003.gguf`) and the quant tag.
pub fn gguf_quant(name: &str) -> Option<String> {
    let stem = name.rsplit('/').next()?.strip_suffix(".gguf")?;
    let stem = match stem.rfind("-of-") {
        Some(i) if stem[..i].rfind('-').is_some() => &stem[..stem[..i].rfind('-').unwrap()],
        _ => stem,
    };
    let up = stem.to_ascii_uppercase();
    const TAGS: &[&str] = &[
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S",
        "IQ3_M", "IQ4_XS", "IQ4_NL", "Q2_K_L", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K_XL",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q4_K_XL", "Q5_0", "Q5_1", "Q5_K_S",
        "Q5_K_M", "Q5_K_L", "Q6_K_L", "Q6_K", "Q8_0", "Q8_K_XL", "BF16", "F16", "F32", "MXFP4",
    ];
    TAGS.iter()
        .filter(|t| {
            up.ends_with(&format!("-{t}"))
                || up.ends_with(&format!(".{t}"))
                || up.ends_with(&format!("_{t}"))
                || up.contains(&format!("-{t}-"))
                || up.contains(&format!(".{t}."))
                || up.contains(&format!("-{t}."))
        })
        .max_by_key(|t| t.len())
        .map(|t| t.to_string())
}

/// Part number and count of a split GGUF (`x-00002-of-00003.gguf`).
pub fn gguf_part(name: &str) -> Option<(u32, u32)> {
    let stem = name.strip_suffix(".gguf")?;
    let i = stem.rfind("-of-")?;
    let total: u32 = stem[i + 4..].parse().ok()?;
    let j = stem[..i].rfind('-')?;
    let n: u32 = stem[j + 1..i].parse().ok()?;
    Some((n, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_tags() {
        assert_eq!(
            gguf_quant("qwen2.5-7b-instruct-q4_k_m.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            gguf_quant("Llama-3.2-1B-Instruct-Q8_0.gguf").as_deref(),
            Some("Q8_0")
        );
        assert_eq!(
            gguf_quant("Q4_K_M/model-Q4_K_M-00001-of-00002.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(gguf_quant("mmproj-model-f16.gguf").as_deref(), Some("F16"));
        assert_eq!(gguf_part("a-00002-of-00003.gguf"), Some((2, 3)));
        assert_eq!(gguf_part("a-q4.gguf"), None);
    }

    #[test]
    fn opaque_plans_on_one_machine() {
        let s = opaque("sdxl", "test", 7 << 30, working_memory(7 << 30));
        assert_eq!(s.num_layers, 1);
        assert!(s.weight_bytes().0 > 7 << 30);
        assert_eq!(s.kv_bytes(0..1, 4096, 4, 2).0, 0);
    }
}
