//! MLX-LM: Apple's MLX on Apple Silicon Macs, one machine per model.

use crate::detect::{python, python_package};
use crate::engine::*;
use crate::util::*;
use crate::{hf, spec};
use anyhow::{bail, Result};
use async_trait::async_trait;

pub struct Mlx;

pub const ID: &str = "mlx";

#[async_trait]
impl Engine for Mlx {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: ID.into(),
            name: "MLX-LM".into(),
            summary: "Apple's MLX: fast on Apple Silicon Macs, runs mlx-community models (4/8-bit) on one Mac.".into(),
            homepage: "https://github.com/ml-explore/mlx-lm".into(),
            tasks: vec![Task::Chat],
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Text],
            multi_machine: MultiMachine::Single,
            security: Security::LocalOnly,
            platforms: "macOS on Apple Silicon".into(),
            model_hint: "An MLX model repo (mlx-community/…) or a local MLX model directory on this server".into(),
            example_model: "mlx-community/Qwen2.5-7B-Instruct-4bit".into(),
            requirements: vec![Requirement::new(
                "mlx-lm",
                "the MLX-LM Python package runs the model",
                &[
                    ("macos", "pip install mlx-lm"),
                    ("linux", "needs a Mac with Apple Silicon"),
                    ("windows", "needs a Mac with Apple Silicon"),
                ],
            )],
        }
    }

    async fn locate(&self) -> Option<Located> {
        let py = python()?;
        let v = python_package(&py, "mlx_lm").await?;
        Some(Located {
            programs: vec![(
                "server".into(),
                vec![
                    py.display().to_string(),
                    "-m".into(),
                    "mlx_lm.server".into(),
                ],
            )],
            version: Some(v),
            ..Default::default()
        })
    }

    async fn detect(&self, host: &HostFacts) -> Detection {
        if !host.apple_silicon {
            let mut d = not_installed(&self.info(), host);
            d.notes.push("MLX runs only on Apple Silicon Macs.".into());
            return d;
        }
        let Some(loc) = self.locate().await else {
            return not_installed(&self.info(), host);
        };
        Detection {
            engine: ID.into(),
            ready: true,
            version: loc.version,
            accelerators: vec!["Metal".into()],
            devices: 1,
            roles: vec!["serve".into()],
            ..Default::default()
        }
    }

    async fn resolve(&self, req: &ModelRequest) -> Result<ResolvedModel> {
        let m = req.model.trim().to_string();
        let path = std::path::Path::new(&m);
        let (spec, local_path) = if path.is_dir() {
            let dir = path.to_path_buf();
            (
                tokio::task::spawn_blocking(move || {
                    tendril_core::model::source::inspect_local_dir(&dir)
                })
                .await??,
                Some(m.clone()),
            )
        } else if hf::is_repo_id(&m) {
            let (repo, rev) = hf::split_repo(&m);
            let files = hf::list_files(&repo, &rev).await?;
            let size: u64 = files
                .iter()
                .filter(|f| f.path.ends_with(".safetensors"))
                .map(|f| f.size)
                .sum();
            if size == 0 {
                bail!("{repo} has no safetensors weights (MLX-LM runs MLX/safetensors models, e.g. from mlx-community)");
            }
            let r = format!("{repo}@{rev}");
            let s = tokio::task::spawn_blocking(move || {
                let r = tendril_core::model::source::resolve(&r)?;
                tendril_core::model::source::inspect(&r, &Default::default())
            })
            .await?;
            // Quantized MLX weights are smaller than the config implies: plan
            // with the real file size.
            let mut s = s.unwrap_or_else(|_| {
                spec::opaque(
                    &repo,
                    "huggingface (size only)",
                    size,
                    spec::working_memory(size),
                )
            });
            let total = s.weight_bytes().0.max(1);
            if s.num_layers > 1 && total != size {
                let f = size as f64 / total as f64;
                let scale = |b: tendril_core::Bytes| tendril_core::Bytes((b.0 as f64 * f) as u64);
                s.bytes.embed = scale(s.bytes.embed);
                s.bytes.lm_head = scale(s.bytes.lm_head);
                s.bytes.layers = s.bytes.layers.iter().map(|b| scale(*b)).collect();
            }
            (s, None)
        } else {
            bail!(
                "MLX-LM needs a HuggingFace repo (e.g. mlx-community/…) or a local model directory"
            );
        };
        Ok(ResolvedModel {
            engine: ID.into(),
            id: m.clone(),
            served_name: m.clone(),
            task: Task::Chat,
            vision: false,
            layered: false,
            notes: vec![format!(
                "{} of weights; MLX-LM downloads them itself from HuggingFace",
                gib(spec.weight_bytes().0)
            )],
            spec,
            files: vec![],
            local_path,
            params: serde_json::json!({}),
        })
    }

    fn roles(&self, m: &ResolvedModel, p: &Placement, _machines: &[Machine]) -> Result<Vec<Role>> {
        let s = single("MLX-LM", p)?;
        let model = m.local_path.clone().unwrap_or_else(|| m.id.clone());
        let l = launch(ID, "server", &[]).args([
            "--model",
            &model,
            "--host",
            "127.0.0.1",
            "--port",
            "{port}",
        ]);
        Ok(vec![role(
            "server",
            &s.machine,
            true,
            Launch {
                ready_timeout: 3600,
                ..l
            },
        )])
    }

    fn transfer_note(&self, _m: &ResolvedModel, _p: &Placement, _stage: usize) -> String {
        "MLX-LM downloads the model from HuggingFace on this Mac".into()
    }
}
