//! Partial checkpoints: exactly the tensors one stage needs, written as a
//! standard safetensors file on the worker and cached for next time.

use crate::proto::TensorEntry;
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tendril_core::model::safetensors::{classify, Component};
use tendril_engine::config::ModelConfig;
use tendril_engine::model::StageSpec;
use tendril_engine::weights::WeightStore;

/// Tensors a stage needs, in a stable order.
pub fn stage_tensors(
    ws: &WeightStore,
    cfg: &ModelConfig,
    spec: &StageSpec,
) -> Result<Vec<TensorEntry>> {
    let head_name = ws.head_name().map(String::from);
    let mut out = Vec::new();
    for name in ws.all_names() {
        let keep = match classify(name) {
            Component::Embed => {
                spec.embed || (spec.head && (cfg.tie_embeddings || head_name.is_none()))
            }
            Component::Layer(i) => i >= spec.layer_start && i < spec.layer_end,
            Component::FinalNorm => spec.head,
            Component::LmHead => spec.head && !cfg.tie_embeddings,
            Component::Ignored => false,
        };
        if keep {
            let (dt, shape, data) = ws.raw(name)?;
            out.push(TensorEntry {
                name: name.clone(),
                dtype: format!("{dt:?}"),
                shape,
                len: data.len() as u64,
            });
        }
    }
    if out.is_empty() {
        bail!("no tensors found for stage {spec:?}");
    }
    Ok(out)
}

/// Identity of a model's weights (not its path): config + tensor table.
pub fn model_key(config_json: &str, ws: &WeightStore) -> Result<String> {
    let mut h = Sha256::new();
    h.update(config_json.as_bytes());
    for n in ws.all_names() {
        let (dt, shape, data) = ws.raw(n)?;
        h.update(format!("{n}:{dt:?}:{shape:?}:{};", data.len()).as_bytes());
    }
    Ok(hex(&h.finalize()[..12]))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn shard_path(cache: &Path, model_key: &str, entries: &[TensorEntry]) -> PathBuf {
    let mut h = Sha256::new();
    for e in entries {
        h.update(format!("{}:{}:{:?}:{};", e.name, e.dtype, e.shape, e.len).as_bytes());
    }
    cache
        .join("shards")
        .join(model_key)
        .join(format!("{}.safetensors", hex(&h.finalize()[..10])))
}

pub fn default_cache() -> PathBuf {
    std::env::var("TENDRIL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::cache_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("tendril")
        })
}

/// A cached shard is complete when its `.ok` marker exists.
pub fn is_complete(path: &Path) -> bool {
    path.with_extension("ok").exists() && path.exists()
}

pub struct ShardWriter {
    file: std::fs::File,
    path: PathBuf,
    data_start: u64,
    offsets: std::collections::HashMap<String, (u64, u64)>,
    written: u64,
    pub total: u64,
}

impl ShardWriter {
    pub fn create(path: &Path, entries: &[TensorEntry]) -> Result<ShardWriter> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let mut header = serde_json::Map::new();
        let mut offsets = std::collections::HashMap::new();
        let mut off = 0u64;
        for e in entries {
            header.insert(
                e.name.clone(),
                serde_json::json!({"dtype": e.dtype, "shape": e.shape, "data_offsets": [off, off + e.len]}),
            );
            offsets.insert(e.name.clone(), (off, e.len));
            off += e.len;
        }
        let mut h = serde_json::to_vec(&header)?;
        while (8 + h.len()) % 8 != 0 {
            h.push(b' ');
        }
        let _ = std::fs::remove_file(path.with_extension("ok"));
        let mut file =
            std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
        file.write_all(&(h.len() as u64).to_le_bytes())?;
        file.write_all(&h)?;
        let data_start = 8 + h.len() as u64;
        file.set_len(data_start + off)?;
        Ok(ShardWriter {
            file,
            path: path.to_path_buf(),
            data_start,
            offsets,
            written: 0,
            total: off,
        })
    }

    pub fn write(&mut self, name: &str, offset: u64, data: &[u8]) -> Result<()> {
        let (start, len) = *self
            .offsets
            .get(name)
            .with_context(|| format!("unexpected tensor {name}"))?;
        if offset + data.len() as u64 > len {
            bail!("tensor {name}: data beyond its declared length");
        }
        self.file
            .seek(SeekFrom::Start(self.data_start + start + offset))?;
        self.file.write_all(data)?;
        self.written += data.len() as u64;
        Ok(())
    }

    pub fn written(&self) -> u64 {
        self.written
    }

    pub fn finish(mut self) -> Result<PathBuf> {
        if self.written != self.total {
            bail!("received {} of {} weight bytes", self.written, self.total);
        }
        self.file.flush()?;
        self.file.sync_all()?;
        std::fs::write(self.path.with_extension("ok"), b"ok")?;
        Ok(self.path)
    }
}
