//! Reading only the tensors a stage needs from safetensors checkpoints.

use anyhow::{bail, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub struct WeightStore {
    st: MmapedSafetensors,
    names: BTreeSet<String>,
    /// Prefix before "layers.N." / "embed_tokens" (e.g. "model.").
    prefix: String,
    head_name: Option<String>,
    pub files: Vec<PathBuf>,
}

impl WeightStore {
    pub fn open(files: &[PathBuf]) -> Result<WeightStore> {
        if files.is_empty() {
            bail!("no .safetensors files found");
        }
        // SAFETY: files are opened read-only; Tendril never mutates model files while serving.
        let st =
            unsafe { MmapedSafetensors::multi(files) }.context("cannot map safetensors files")?;
        let names: BTreeSet<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
        // The prefix before "layers.N." / "embed_tokens" / "norm.weight"; partial
        // checkpoints (one pipeline stage) may contain any subset of these.
        let prefix = names
            .iter()
            .find_map(|n| {
                if let Some(i) = n.find("layers.") {
                    let rest = &n[i + 7..];
                    if rest.chars().next().is_some_and(|c| c.is_ascii_digit())
                        && !n[..i].contains("vision")
                    {
                        return Some(n[..i].to_string());
                    }
                }
                None
            })
            .or_else(|| {
                names
                    .iter()
                    .find_map(|n| n.strip_suffix("embed_tokens.weight").map(String::from))
            })
            .or_else(|| {
                names.iter().find_map(|n| {
                    n.strip_suffix("norm.weight")
                        .filter(|p| !p.contains("layers"))
                        .map(String::from)
                })
            })
            .unwrap_or_else(|| "model.".to_string());
        let head_name = [
            "lm_head.weight",
            "language_model.lm_head.weight",
            "model.lm_head.weight",
        ]
        .into_iter()
        .find(|n| names.contains(*n))
        .map(String::from);
        Ok(WeightStore {
            st,
            names,
            prefix,
            head_name,
            files: files.to_vec(),
        })
    }

    pub fn open_dir(dir: &Path) -> Result<WeightStore> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("read {}", dir.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        Self::open(&files)
    }

    pub fn name(&self, local: &str) -> String {
        format!("{}{}", self.prefix, local)
    }

    pub fn has(&self, full: &str) -> bool {
        self.names.contains(full)
    }

    pub fn head_name(&self) -> Option<&str> {
        self.head_name.as_deref()
    }

    /// Load a tensor onto `device` converted to `dtype`.
    pub fn tensor(&self, full: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let t = self
            .st
            .load(full, &Device::Cpu)
            .with_context(|| format!("missing tensor {full}"))?;
        let t = if t.dtype() != dtype {
            t.to_dtype(dtype)?
        } else {
            t
        };
        Ok(t.to_device(device)?)
    }

    /// Raw view: (dtype, shape, bytes) without conversion.
    pub fn raw(&self, full: &str) -> Result<(safetensors::Dtype, Vec<usize>, &[u8])> {
        let v = self
            .st
            .get(full)
            .with_context(|| format!("missing tensor {full}"))?;
        Ok((v.dtype(), v.shape().to_vec(), v.data()))
    }

    pub fn all_names(&self) -> impl Iterator<Item = &String> {
        self.names.iter()
    }
}
