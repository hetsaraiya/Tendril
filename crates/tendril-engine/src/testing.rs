//! Tiny random-weight models for tests and demos (no downloads needed).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

fn byte_chars() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = vec!['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap();
    }
    out
}

/// Write a byte-level tokenizer (ids 0–4 special, 5–260 bytes) with a ChatML template.
pub fn write_tokenizer(dir: &Path) -> Result<()> {
    let specials = ["<pad>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"];
    let mut vocab = serde_json::Map::new();
    for (i, s) in specials.iter().enumerate() {
        vocab.insert(s.to_string(), json!(i));
    }
    for (b, c) in byte_chars().into_iter().enumerate() {
        vocab.insert(c.to_string(), json!(5 + b));
    }
    let added: Vec<_> = specials
        .iter()
        .enumerate()
        .map(|(i, s)| json!({"id": i, "content": s, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
        .collect();
    let tok = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": vocab, "merges": []}
    });
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tok)?)?;
    let tc = json!({
        "bos_token": "<s>", "eos_token": "<|im_end|>", "pad_token": "<pad>", "add_bos_token": false,
        "chat_template": crate::tokenizer::CHATML,
    });
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_vec_pretty(&tc)?,
    )?;
    Ok(())
}

/// Write a random Llama-architecture model: config.json, model.safetensors, tokenizer.
pub fn write_tiny_llama(
    dir: &Path,
    layers: usize,
    hidden: usize,
    seed: u64,
    dtype: DType,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let vocab = 384usize;
    let heads = 4usize;
    let kv = 2usize;
    let hd = hidden / heads;
    let inter = hidden * 2;
    let cfg = json!({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": vocab, "hidden_size": hidden, "intermediate_size": inter,
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv,
        "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 4,
        "torch_dtype": match dtype { DType::BF16 => "bfloat16", DType::F16 => "float16", _ => "float32" },
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&cfg)?)?;
    let dev = Device::Cpu;
    use rand::{Rng, SeedableRng};
    let rng = std::cell::RefCell::new(rand::rngs::StdRng::seed_from_u64(seed));
    let normal = |n: usize, std: f32, mean: f32| -> Vec<f32> {
        let mut g = rng.borrow_mut();
        (0..n)
            .map(|_| {
                let u1: f32 = g.random::<f32>().max(1e-7);
                let u2: f32 = g.random::<f32>();
                mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect()
    };
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let r = |shape: (usize, usize), std: f32| -> Result<Tensor> {
        Ok(Tensor::from_vec(normal(shape.0 * shape.1, std, 0.0), shape, &dev)?.to_dtype(dtype)?)
    };
    let ones = |n: usize| -> Result<Tensor> {
        Ok(Tensor::from_vec(normal(n, 0.1, 1.0), n, &dev)?.to_dtype(dtype)?)
    };
    t.insert("model.embed_tokens.weight".into(), r((vocab, hidden), 1.0)?);
    for i in 0..layers {
        let p = format!("model.layers.{i}");
        t.insert(
            format!("{p}.self_attn.q_proj.weight"),
            r((heads * hd, hidden), 0.08)?,
        );
        t.insert(
            format!("{p}.self_attn.k_proj.weight"),
            r((kv * hd, hidden), 0.08)?,
        );
        t.insert(
            format!("{p}.self_attn.v_proj.weight"),
            r((kv * hd, hidden), 0.08)?,
        );
        t.insert(
            format!("{p}.self_attn.o_proj.weight"),
            r((hidden, heads * hd), 0.08)?,
        );
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            r((inter, hidden), 0.08)?,
        );
        t.insert(format!("{p}.mlp.up_proj.weight"), r((inter, hidden), 0.08)?);
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            r((hidden, inter), 0.08)?,
        );
        t.insert(format!("{p}.input_layernorm.weight"), ones(hidden)?);
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            ones(hidden)?,
        );
    }
    t.insert("model.norm.weight".into(), ones(hidden)?);
    t.insert("lm_head.weight".into(), r((vocab, hidden), 0.1)?);
    candle_core::safetensors::save(&t, dir.join("model.safetensors"))?;
    write_tokenizer(dir)?;
    Ok(())
}
