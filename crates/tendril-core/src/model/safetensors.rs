//! Safetensors header parsing and component accounting.

use super::ComponentBytes;
use crate::units::Bytes;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

/// Headers larger than this are rejected before allocation.
pub const MAX_HEADER: u64 = 100 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Byte range relative to the start of the data section.
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Parse the JSON header (without the 8-byte length prefix).
pub fn parse_header(json: &[u8]) -> Result<Vec<TensorInfo>> {
    let v: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(json).context("invalid safetensors header JSON")?;
    let mut out = Vec::with_capacity(v.len());
    for (name, t) in v {
        if name == "__metadata__" {
            continue;
        }
        let dtype = t
            .get("dtype")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        let shape = t
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        let offs = t
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .context("tensor without data_offsets")?;
        if offs.len() != 2 {
            bail!("tensor {name}: malformed data_offsets");
        }
        let start = offs[0].as_u64().context("bad offset")?;
        let end = offs[1].as_u64().context("bad offset")?;
        if end < start {
            bail!("tensor {name}: end offset before start");
        }
        out.push(TensorInfo {
            name,
            dtype,
            shape,
            start,
            end,
        });
    }
    Ok(out)
}

/// Read a local safetensors header. Returns tensors and the data-section offset.
pub fn read_local_header(path: &Path) -> Result<(Vec<TensorInfo>, u64)> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .context("file too short for a safetensors header")?;
    let n = u64::from_le_bytes(len);
    if n > MAX_HEADER {
        bail!(
            "{}: safetensors header of {n} bytes is implausibly large",
            path.display()
        );
    }
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf)
        .context("truncated safetensors header")?;
    Ok((parse_header(&buf)?, 8 + n))
}

/// Which model component a tensor belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    Embed,
    Layer(usize),
    FinalNorm,
    LmHead,
    /// Vision towers, projectors, MTP heads... not part of the text decoder.
    Ignored,
}

pub fn classify(name: &str) -> Component {
    let n = name;
    if n.contains("vision")
        || n.contains("multi_modal")
        || n.contains("audio")
        || n.contains("mm_projector")
    {
        return Component::Ignored;
    }
    for marker in [".layers.", ".h.", "blk.", "layers."] {
        if let Some(pos) = n.find(marker) {
            let rest = &n[pos + marker.len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(i) = digits.parse() {
                return Component::Layer(i);
            }
        }
    }
    if n.contains("embed_tokens")
        || n.contains("wte")
        || n.contains("tok_embeddings")
        || n.starts_with("token_embd")
    {
        return Component::Embed;
    }
    if n.starts_with("lm_head")
        || n.contains(".lm_head")
        || n == "output.weight"
        || n.starts_with("output.")
    {
        return Component::LmHead;
    }
    if n.ends_with("norm.weight")
        || n.contains("ln_f")
        || n.starts_with("output_norm")
        || n.ends_with("norm.bias")
    {
        return Component::FinalNorm;
    }
    Component::Ignored
}

/// Sum tensor bytes into components.
pub fn component_bytes(tensors: &[TensorInfo], num_layers: usize) -> ComponentBytes {
    let mut c = ComponentBytes {
        layers: vec![Bytes::ZERO; num_layers],
        ..Default::default()
    };
    for t in tensors {
        let b = Bytes(t.len());
        c.largest_tensor = c.largest_tensor.max(b);
        match classify(&t.name) {
            Component::Embed => c.embed += b,
            Component::Layer(i) if i < num_layers => c.layers[i] += b,
            Component::Layer(_) => {}
            Component::FinalNorm => c.final_norm += b,
            Component::LmHead => c.lm_head += b,
            Component::Ignored => {}
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_names() {
        assert_eq!(classify("model.embed_tokens.weight"), Component::Embed);
        assert_eq!(
            classify("model.layers.12.self_attn.q_proj.weight"),
            Component::Layer(12)
        );
        assert_eq!(classify("model.norm.weight"), Component::FinalNorm);
        assert_eq!(classify("lm_head.weight"), Component::LmHead);
        assert_eq!(classify("blk.3.attn_q.weight"), Component::Layer(3));
        assert_eq!(classify("token_embd.weight"), Component::Embed);
        assert_eq!(classify("output_norm.weight"), Component::FinalNorm);
        assert_eq!(classify("output.weight"), Component::LmHead);
        assert_eq!(
            classify("language_model.model.layers.0.mlp.up_proj.weight"),
            Component::Layer(0)
        );
        assert_eq!(
            classify("vision_tower.encoder.layers.0.x"),
            Component::Ignored
        );
    }

    #[test]
    fn header_roundtrip() {
        let h = br#"{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let t = parse_header(h).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].len(), 12);
        assert!(
            parse_header(br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#).is_err()
        );
    }
}
