//! Minimal GGUF reader: metadata and tensor table, no tensor data.

use super::safetensors::{classify, Component};
use super::{Arch, Attention, ComponentBytes, ModelSpec, ParamCounts, Quant};
use crate::units::Bytes;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MetaValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    /// Arrays are summarized: length and (for small arrays) values.
    Array(u64, Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::Int(i) if *i >= 0 => Some(*i as u64),
            MetaValue::Float(f) if *f >= 0.0 => Some(*f as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub meta: BTreeMap<String, MetaValue>,
    pub tensors: Vec<GgufTensor>,
    /// Absolute file offset where tensor data begins.
    pub data_start: u64,
}

const MAX_STR: u64 = 16 * 1024 * 1024;
const MAX_ITEMS: u64 = 1 << 24;

struct CountingReader<R> {
    inner: R,
    pos: u64,
}
impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

fn rd<const N: usize, R: Read>(r: &mut R) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)
        .context("unexpected end of GGUF header")?;
    Ok(b)
}
fn rd_u32<R: Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(rd::<4, _>(r)?))
}
fn rd_u64<R: Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(rd::<8, _>(r)?))
}
fn rd_str<R: Read>(r: &mut R) -> Result<String> {
    let n = rd_u64(r)?;
    if n > MAX_STR {
        bail!("GGUF string of {n} bytes is implausibly long");
    }
    let mut b = vec![0u8; n as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn rd_value<R: Read>(r: &mut R, ty: u32, depth: u32) -> Result<MetaValue> {
    Ok(match ty {
        0 => MetaValue::Int(rd::<1, _>(r)?[0] as i64),
        1 => MetaValue::Int(rd::<1, _>(r)?[0] as i8 as i64),
        2 => MetaValue::Int(u16::from_le_bytes(rd::<2, _>(r)?) as i64),
        3 => MetaValue::Int(i16::from_le_bytes(rd::<2, _>(r)?) as i64),
        4 => MetaValue::Int(rd_u32(r)? as i64),
        5 => MetaValue::Int(rd_u32(r)? as i32 as i64),
        6 => MetaValue::Float(f32::from_le_bytes(rd::<4, _>(r)?) as f64),
        7 => MetaValue::Bool(rd::<1, _>(r)?[0] != 0),
        8 => MetaValue::Str(rd_str(r)?),
        9 => {
            if depth > 2 {
                bail!("GGUF arrays nested too deeply");
            }
            let ety = rd_u32(r)?;
            let n = rd_u64(r)?;
            if n > MAX_ITEMS {
                bail!("GGUF array of {n} items is implausibly long");
            }
            let mut keep = Vec::new();
            for i in 0..n {
                let v = rd_value(r, ety, depth + 1)?;
                if i < 64 {
                    keep.push(v);
                }
            }
            MetaValue::Array(n, keep)
        }
        10 => MetaValue::Int(rd_u64(r)? as i64),
        11 => MetaValue::Int(rd_u64(r)? as i64),
        12 => MetaValue::Float(f64::from_le_bytes(rd::<8, _>(r)?)),
        _ => bail!("unknown GGUF metadata type {ty}"),
    })
}

/// (block elements, block bytes) for ggml tensor types.
pub fn ggml_block(ty: u32) -> Option<(u64, u64)> {
    Some(match ty {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        30 => (1, 2),
        _ => return None,
    })
}

pub fn ggml_type_name(ty: u32) -> &'static str {
    match ty {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        30 => "BF16",
        _ => "other",
    }
}

pub fn read_header<R: Read>(r: R, file_len: Option<u64>) -> Result<GgufHeader> {
    let mut r = CountingReader { inner: r, pos: 0 };
    let magic = rd::<4, _>(&mut r)?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file (bad magic)");
    }
    let version = rd_u32(&mut r)?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version}");
    }
    let n_tensors = rd_u64(&mut r)?;
    let n_kv = rd_u64(&mut r)?;
    if n_tensors > MAX_ITEMS || n_kv > MAX_ITEMS {
        bail!("GGUF header counts are implausible");
    }
    let mut meta = BTreeMap::new();
    for _ in 0..n_kv {
        let k = rd_str(&mut r)?;
        let ty = rd_u32(&mut r)?;
        let v = rd_value(&mut r, ty, 0)?;
        meta.insert(k, v);
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = rd_str(&mut r)?;
        let nd = rd_u32(&mut r)?;
        if nd > 8 {
            bail!("tensor {name} has {nd} dimensions");
        }
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(rd_u64(&mut r)?);
        }
        let ggml_type = rd_u32(&mut r)?;
        let offset = rd_u64(&mut r)?;
        tensors.push(GgufTensor {
            name,
            dims,
            ggml_type,
            offset,
            size: 0,
        });
    }
    let align = meta
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32)
        .max(1);
    let data_start = r.pos.div_ceil(align) * align;

    // Sizes: from the type table, falling back to offset differences.
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&i| tensors[i].offset);
    for (k, &i) in order.iter().enumerate() {
        let t = &tensors[i];
        let n: u64 = t.dims.iter().product();
        let by_type = ggml_block(t.ggml_type).map(|(be, bb)| n.div_ceil(be) * bb);
        let by_offset = match order.get(k + 1) {
            Some(&j) => Some(tensors[j].offset - t.offset),
            None => file_len.map(|l| l.saturating_sub(data_start + t.offset)),
        };
        tensors[i].size = by_type.or(by_offset).unwrap_or(0);
    }
    Ok(GgufHeader {
        version,
        meta,
        tensors,
        data_start,
    })
}

impl GgufHeader {
    fn arch_key(&self, suffix: &str) -> Option<u64> {
        let arch = self.meta.get("general.architecture")?.as_str()?;
        self.meta.get(&format!("{arch}.{suffix}"))?.as_u64()
    }

    pub fn to_spec(&self, id: &str) -> Result<ModelSpec> {
        let arch_name = self
            .meta
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let arch = Arch::from_hf(match arch_name.as_str() {
            "gemma3" => "gemma3",
            other => other,
        });
        let layers = self
            .arch_key("block_count")
            .context("GGUF missing block_count")?;
        let hidden = self
            .arch_key("embedding_length")
            .context("GGUF missing embedding_length")?;
        let heads = self.arch_key("attention.head_count").unwrap_or(1).max(1);
        let kv = self.arch_key("attention.head_count_kv").unwrap_or(heads);
        let head_dim = self
            .arch_key("attention.key_length")
            .unwrap_or(hidden / heads);
        let inter = self.arch_key("feed_forward_length").unwrap_or(4 * hidden);
        let ctx = self.arch_key("context_length").unwrap_or(4096);
        let vocab = match self.meta.get("tokenizer.ggml.tokens") {
            Some(MetaValue::Array(n, _)) => *n,
            _ => self
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .and_then(|t| t.dims.get(1).copied())
                .unwrap_or(32000),
        };
        let window = self.arch_key("attention.sliding_window");
        let mut attention = vec![Attention::Global; layers as usize];
        if let Some(w) = window {
            for (i, a) in attention.iter_mut().enumerate() {
                let sliding = match arch {
                    Arch::Gemma2 => i % 2 == 0,
                    Arch::Gemma3 => (i + 1) % 6 != 0,
                    _ => true,
                };
                if sliding {
                    *a = Attention::Sliding(w);
                }
            }
        }

        let mut bytes = ComponentBytes {
            layers: vec![Bytes::ZERO; layers as usize],
            ..Default::default()
        };
        let mut params = ParamCounts {
            layers: vec![0; layers as usize],
            layer_small: vec![0; layers as usize],
            ..Default::default()
        };
        let mut type_bytes: BTreeMap<u32, u64> = BTreeMap::new();
        for t in &self.tensors {
            let n: u64 = t.dims.iter().product();
            let b = Bytes(t.size);
            bytes.largest_tensor = bytes.largest_tensor.max(b);
            *type_bytes.entry(t.ggml_type).or_default() += t.size;
            match classify(&t.name) {
                Component::Embed => {
                    bytes.embed += b;
                    params.embed += n;
                }
                Component::Layer(i) if i < layers as usize => {
                    bytes.layers[i] += b;
                    params.layers[i] += n;
                    if t.dims.len() == 1 {
                        params.layer_small[i] += n;
                    }
                }
                Component::FinalNorm => {
                    bytes.final_norm += b;
                    params.final_norm += n;
                }
                Component::LmHead => {
                    bytes.lm_head += b;
                    params.lm_head += n;
                }
                _ => {}
            }
        }
        let tie = bytes.lm_head.0 == 0;
        // Dominant storage type decides the label.
        let dominant = type_bytes
            .iter()
            .max_by_key(|(_, b)| **b)
            .map(|(t, _)| *t)
            .unwrap_or(1);
        let stored = match dominant {
            0 => Quant::F32,
            1 => Quant::F16,
            30 => Quant::Bf16,
            8 => Quant::Q8_0,
            14 => Quant::Q6K,
            13 => Quant::Q5K,
            12 => Quant::Q4K,
            2 => Quant::Q4_0,
            _ => Quant::Mixed,
        };
        let mut notes = vec![format!(
            "GGUF v{}: {}",
            self.version,
            type_bytes
                .iter()
                .map(|(t, b)| format!("{} {}", ggml_type_name(*t), Bytes(*b)))
                .collect::<Vec<_>>()
                .join(", ")
        )];
        if !arch.executable() {
            notes.push(format!(
                "Architecture '{arch_name}' is not supported by Tendril's engine yet."
            ));
        }
        Ok(ModelSpec {
            id: id.to_string(),
            origin: "gguf".into(),
            arch,
            model_type: arch_name,
            num_layers: layers,
            hidden_size: hidden,
            intermediate_size: inter,
            num_heads: heads,
            num_kv_heads: kv,
            head_dim,
            vocab_size: vocab,
            max_position: ctx,
            tie_embeddings: tie,
            attention,
            params,
            stored,
            repr: stored,
            bytes,
            bytes_measured: true,
            moe: None,
            notes,
        })
    }
}

/// Serialize a GGUF header (used by tests and by the engine's test fixtures).
pub mod write {
    pub fn string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    pub fn kv_u32(out: &mut Vec<u8>, k: &str, v: u32) {
        string(out, k);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    pub fn kv_str(out: &mut Vec<u8>, k: &str, v: &str) {
        string(out, k);
        out.extend_from_slice(&8u32.to_le_bytes());
        string(out, v);
    }
}

#[cfg(test)]
mod tests {
    use super::write::*;
    use super::*;

    #[test]
    fn parse_synthetic() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes()); // tensors
        b.extend_from_slice(&5u64.to_le_bytes()); // kv
        kv_str(&mut b, "general.architecture", "llama");
        kv_u32(&mut b, "llama.block_count", 1);
        kv_u32(&mut b, "llama.embedding_length", 256);
        kv_u32(&mut b, "llama.attention.head_count", 4);
        kv_u32(&mut b, "llama.feed_forward_length", 512);
        let tensor = |b: &mut Vec<u8>, name: &str, dims: &[u64], ty: u32, off: u64| {
            string(b, name);
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&off.to_le_bytes());
        };
        tensor(&mut b, "token_embd.weight", &[256, 1000], 8, 0);
        tensor(&mut b, "blk.0.attn_q.weight", &[256, 256], 12, 272000);
        tensor(&mut b, "output_norm.weight", &[256], 0, 272000 + 36864);
        let h = read_header(&b[..], None).unwrap();
        assert_eq!(h.tensors.len(), 3);
        assert_eq!(h.tensors[0].size, 256 * 1000 / 32 * 34);
        assert_eq!(h.tensors[1].size, 256 * 256 / 256 * 144);
        let s = h.to_spec("t").unwrap();
        assert_eq!(s.num_layers, 1);
        assert!(s.tie_embeddings);
        assert_eq!(s.vocab_size, 1000);
        assert!(s.bytes_measured);
    }
}
