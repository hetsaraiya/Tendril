//! exo: forms its own cluster of Macs (and Linux CPUs) and decides the split.

use crate::detect::{find_program, python, python_package};
use crate::engine::*;
use crate::util::*;
use crate::{hf, spec};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use std::time::{Duration, Instant};

pub struct Exo;

pub const ID: &str = "exo";

#[async_trait]
impl Engine for Exo {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: ID.into(),
            name: "exo".into(),
            summary: "Pools Apple Silicon Macs (MLX, RDMA over Thunderbolt) into one cluster and decides the split itself.".into(),
            homepage: "https://github.com/exo-explore/exo".into(),
            tasks: vec![Task::Chat],
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Text],
            multi_machine: MultiMachine::EngineManaged,
            security: Security::TrustedNetworkOnly {
                what: "exo finds its peers and moves activations with its own networking (zenoh discovery, MLX distributed), which Tendril cannot wrap in its encryption. Tendril isolates each deployment with a namespace derived from the cluster token.".into(),
                risk: "exo's traffic between machines is not encrypted, and a namespace only keeps honest clusters apart — it is not a password. Use only on a private network you trust.".into(),
            },
            platforms: "macOS on Apple Silicon (Linux runs on the CPU)".into(),
            model_hint: "An exo model id (llama-3.2-1b) or an MLX repo (mlx-community/…)".into(),
            example_model: "mlx-community/Llama-3.2-3B-Instruct-4bit".into(),
            requirements: vec![Requirement::new(
                "exo",
                "runs the model and links the machines",
                &[
                    ("macos", "brew install --cask exo (or: git clone https://github.com/exo-explore/exo && cd exo && uv sync)"),
                    ("linux", "git clone https://github.com/exo-explore/exo && cd exo && uv sync --extra mlx-cpu"),
                    ("windows", "not supported"),
                ],
            )],
        }
    }

    async fn locate(&self) -> Option<Located> {
        if let Some(p) = find_program(&["exo"]) {
            return Some(Located {
                programs: vec![("server".into(), vec![p.display().to_string()])],
                ..Default::default()
            });
        }
        // A source checkout run with uv.
        let dirs = [
            std::env::var("TENDRIL_EXO_DIR")
                .ok()
                .map(std::path::PathBuf::from),
            dirs::home_dir().map(|h| h.join("exo")),
        ];
        if let (Some(dir), Some(uv)) = (
            dirs.into_iter()
                .flatten()
                .find(|d| d.join("pyproject.toml").exists() && d.join("src/exo").exists()),
            find_program(&["uv"]),
        ) {
            return Some(Located {
                programs: vec![(
                    "server".into(),
                    vec![uv.display().to_string(), "run".into(), "exo".into()],
                )],
                dir: Some(dir),
                ..Default::default()
            });
        }
        let py = python()?;
        let v = python_package(&py, "exo").await?;
        Some(Located {
            programs: vec![(
                "server".into(),
                vec![py.display().to_string(), "-m".into(), "exo.main".into()],
            )],
            version: Some(v),
            ..Default::default()
        })
    }

    async fn detect(&self, host: &HostFacts) -> Detection {
        let Some(loc) = self.locate().await else {
            return not_installed(&self.info(), host);
        };
        let (accelerators, devices) = if host.apple_silicon {
            (vec!["Metal".into()], 1)
        } else {
            (vec!["CPU".into()], 0)
        };
        let mut d = Detection {
            engine: ID.into(),
            ready: true,
            version: loc.version,
            accelerators,
            devices,
            roles: vec!["serve".into(), "split-worker".into()],
            ..Default::default()
        };
        if !host.apple_silicon {
            d.notes
                .push("exo runs on the CPU here (its GPU support is Apple Silicon only).".into());
        }
        d
    }

    async fn resolve(&self, req: &ModelRequest) -> Result<ResolvedModel> {
        let m = req.model.trim().to_string();
        if m.is_empty() {
            bail!("empty model name");
        }
        let (size, note) = if hf::is_repo_id(&m) {
            let (repo, rev) = hf::split_repo(&m);
            let files = hf::list_files(&repo, &rev).await?;
            let s: u64 = files
                .iter()
                .filter(|f| f.path.ends_with(".safetensors"))
                .map(|f| f.size)
                .sum();
            if s == 0 {
                bail!("{repo} has no safetensors weights (exo runs MLX models, e.g. from mlx-community)");
            }
            (s, format!("{} of weights", gib(s)))
        } else {
            (
                0,
                "Size unknown until exo places it (exo model card)".into(),
            )
        };
        Ok(ResolvedModel {
            engine: ID.into(),
            id: m.clone(),
            served_name: m.clone(),
            task: Task::Chat,
            vision: false,
            layered: false,
            spec: spec::opaque(&m, "exo", size, 0),
            files: vec![],
            local_path: None,
            params: serde_json::json!({ "size": size }),
            notes: vec![
                note,
                "exo downloads the model and decides how to split it across the machines it finds"
                    .into(),
            ],
        })
    }

    fn arrange(
        &self,
        _m: &ResolvedModel,
        mut p: Placement,
        _machines: &[Machine],
    ) -> Result<Placement> {
        p.head = 0;
        Ok(p)
    }

    fn roles(&self, m: &ResolvedModel, p: &Placement, _machines: &[Machine]) -> Result<Vec<Role>> {
        let n = p.stages.len();
        let mut roles = Vec::new();
        for (i, s) in p.stages.iter().enumerate() {
            let mut l = launch(ID, "server", &[])
                .args(["--namespace", "{namespace}"])
                .env("EXO_ZENOH_NAMESPACE", "{namespace}");
            l.lan = n > 1;
            if i == 0 {
                l = l.args(["--force-master", "--api-port", "{port}"]);
                l.ready = Ready::Http {
                    path: "/node_id".into(),
                };
                l.ready_timeout = 300;
                l.extra = serde_json::json!({ "model": m.id, "nodes": n });
                roles.push(role("server", &s.machine, true, l));
            } else {
                l = l.arg("--no-api");
                l.ready = Ready::Started;
                l.ready_timeout = 60;
                let mut r = role(&format!("node-{i}"), &s.machine, false, l);
                r.after_started = vec!["server".into()];
                roles.push(r);
            }
        }
        Ok(roles)
    }

    /// Ask exo to place the model on the machines it found, then wait for it.
    async fn prepare(&self, launch: &Launch, base: &str, progress: Progress) -> Result<()> {
        let model = launch.extra["model"]
            .as_str()
            .context("no model")?
            .to_string();
        let nodes = launch.extra["nodes"].as_u64().unwrap_or(1);
        let c = reqwest::Client::new();
        let t0 = Instant::now();
        // Peers take a moment to find each other; placement fails until then.
        loop {
            progress("waiting for exo peers", 0, 0);
            let r = c
                .post(format!("{base}/place_instance"))
                .json(&serde_json::json!({ "model_id": model, "min_nodes": nodes }))
                .send()
                .await?;
            if r.status().is_success() {
                break;
            }
            let why = r.text().await.unwrap_or_default();
            if t0.elapsed() > Duration::from_secs(180) {
                bail!("exo could not place {model} on {nodes} machine(s): {why}");
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
        progress("exo is downloading and loading the model", 0, 0);
        let resp = c
            .get(format!("{base}/instance/await"))
            .query(&[("model_id", model.as_str())])
            .send()
            .await?;
        let mut body = resp.bytes_stream();
        let mut buf = String::new();
        while let Some(chunk) = body.next().await {
            buf.push_str(&String::from_utf8_lossy(&chunk?));
            while let Some(i) = buf.find('\n') {
                let line: String = buf.drain(..=i).collect();
                if let Some(data) = line.trim().strip_prefix("data:") {
                    let v: serde_json::Value =
                        serde_json::from_str(data.trim()).unwrap_or_default();
                    match v["type"].as_str() {
                        Some("ready") => return Ok(()),
                        Some("timeout") => {
                            bail!("exo: {}", v["message"].as_str().unwrap_or("timed out"))
                        }
                        _ => {}
                    }
                }
            }
        }
        bail!("exo stopped before {model} was ready")
    }

    fn transfer_note(&self, _m: &ResolvedModel, _p: &Placement, _stage: usize) -> String {
        "exo downloads the part it runs here".into()
    }
}
