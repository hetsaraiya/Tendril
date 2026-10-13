//! Ollama: one-command model management (`llama3.2:3b`), one machine per model.

use crate::detect::{find_program, first_version, run_capture};
use crate::engine::*;
use crate::util::*;
use crate::{hf, spec};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use std::time::Duration;

pub struct Ollama;

pub const ID: &str = "ollama";

/// Rough KV/working memory per token of context (a 7–8B GQA model at f16).
const KV_PER_TOKEN: u64 = 128 << 10;

#[async_trait]
impl Engine for Ollama {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: ID.into(),
            name: "Ollama".into(),
            summary: "Easiest model management: models by name from the Ollama library (or hf.co/…); runs each model on one machine.".into(),
            homepage: "https://ollama.com".into(),
            tasks: vec![Task::Chat],
            inputs: vec![Modality::Text, Modality::Image],
            outputs: vec![Modality::Text],
            multi_machine: MultiMachine::Single,
            security: Security::LocalOnly,
            platforms: "macOS, Linux, Windows".into(),
            model_hint: "An Ollama model name (llama3.2:3b, qwen2.5:7b) or hf.co/org/repo:QUANT".into(),
            example_model: "llama3.2:3b".into(),
            requirements: vec![Requirement::new(
                "ollama",
                "runs and downloads the model",
                &[
                    ("macos", "brew install ollama"),
                    ("linux", "curl -fsSL https://ollama.com/install.sh | sh"),
                    ("windows", "winget install Ollama.Ollama"),
                ],
            )],
        }
    }

    async fn locate(&self) -> Option<Located> {
        let p = find_program(&["ollama"])?;
        Some(Located {
            programs: vec![(
                "server".into(),
                vec![p.display().to_string(), "serve".into()],
            )],
            ..Default::default()
        })
    }

    async fn detect(&self, host: &HostFacts) -> Detection {
        let Some(loc) = self.locate().await else {
            return not_installed(&self.info(), host);
        };
        let bin = loc.program("server").unwrap()[0].clone();
        let version = run_capture(&[bin, "--version".into()], Duration::from_secs(10))
            .await
            .and_then(|o| {
                o.lines()
                    .find(|l| l.contains("version"))
                    .map(|l| l.rsplit(' ').next().unwrap_or(l).to_string())
                    .or_else(|| first_version(&o))
            });
        let (accelerators, devices) = host_accelerators(host);
        Detection {
            engine: ID.into(),
            ready: true,
            version,
            accelerators,
            devices,
            roles: vec!["serve".into()],
            ..Default::default()
        }
    }

    async fn resolve(&self, req: &ModelRequest) -> Result<ResolvedModel> {
        let name = req.model.trim().trim_start_matches("ollama://").to_string();
        if name.is_empty() || name.contains(' ') {
            bail!("'{name}' is not an Ollama model name");
        }
        let (size, vision, note) = match size_of(&name).await {
            Ok((s, v)) => (s, v, None),
            Err(e) => {
                let guess = guess_size(&name).with_context(|| {
                    format!("cannot find {name} in the Ollama library ({e:#}) — check the name on ollama.com/library")
                })?;
                (
                    guess,
                    false,
                    Some(format!("Size guessed from the name ({e:#}).")),
                )
            }
        };
        let kv = KV_PER_TOKEN * req.context.max(512) as u64 * req.concurrency.max(1) as u64;
        let spec = spec::opaque(&name, "ollama", size, spec::working_memory(size) + kv);
        let mut notes = vec![format!("{} download", gib(size))];
        notes.extend(note);
        Ok(ResolvedModel {
            engine: ID.into(),
            id: name.clone(),
            served_name: name.clone(),
            task: Task::Chat,
            vision,
            layered: false,
            spec,
            files: vec![],
            local_path: None,
            params: serde_json::json!({ "size": size }),
            notes,
        })
    }

    fn roles(&self, m: &ResolvedModel, p: &Placement, _machines: &[Machine]) -> Result<Vec<Role>> {
        let s = single("Ollama", p)?;
        let mut l = launch(ID, "server", &[])
            .env("OLLAMA_HOST", "127.0.0.1:{port}")
            .env("OLLAMA_CONTEXT_LENGTH", p.context.max(512).to_string())
            .env("OLLAMA_NUM_PARALLEL", p.concurrency.max(1).to_string())
            .env("OLLAMA_KEEP_ALIVE", "-1")
            .env("OLLAMA_MAX_LOADED_MODELS", "1")
            .ready(Ready::Http {
                path: "/api/version".into(),
            });
        l.extra = serde_json::json!({ "model": m.served_name });
        l.ready_timeout = 120;
        Ok(vec![role("server", &s.machine, true, l)])
    }

    /// Pull the model (with progress), then load it so the first request is fast.
    async fn prepare(&self, launch: &Launch, base: &str, progress: Progress) -> Result<()> {
        let model = launch.extra["model"]
            .as_str()
            .context("no model")?
            .to_string();
        let c = reqwest::Client::new();
        let resp = c
            .post(format!("{base}/api/pull"))
            .json(&serde_json::json!({ "model": model, "stream": true }))
            .send()
            .await?;
        if !resp.status().is_success() {
            bail!("ollama pull {model}: HTTP {}", resp.status());
        }
        let mut body = resp.bytes_stream();
        let mut buf = Vec::new();
        while let Some(chunk) = body.next().await {
            buf.extend_from_slice(&chunk?);
            while let Some(i) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=i).collect();
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&line) else {
                    continue;
                };
                if let Some(e) = v.get("error").and_then(|e| e.as_str()) {
                    bail!("ollama pull {model}: {e}");
                }
                let total = v["total"].as_u64().unwrap_or(0);
                let done = v["completed"].as_u64().unwrap_or(0);
                progress(v["status"].as_str().unwrap_or("pulling"), done, total);
            }
        }
        progress("loading", 0, 0);
        let r = c
            .post(format!("{base}/api/generate"))
            .json(&serde_json::json!({ "model": model, "keep_alive": -1 }))
            .send()
            .await?;
        if !r.status().is_success() {
            bail!(
                "ollama could not load {model}: {}",
                r.text().await.unwrap_or_default()
            );
        }
        Ok(())
    }

    fn transfer_note(&self, m: &ResolvedModel, _p: &Placement, _stage: usize) -> String {
        format!(
            "ollama pulls {} ({}) unless it already has it",
            m.id,
            gib(m.params["size"].as_u64().unwrap_or(0))
        )
    }
}

/// Download size of an Ollama model and whether it has a vision projector.
async fn size_of(name: &str) -> Result<(u64, bool)> {
    if let Some(rest) = name
        .strip_prefix("hf.co/")
        .or_else(|| name.strip_prefix("huggingface.co/"))
    {
        let (repo, quant) = match rest.rsplit_once(':') {
            Some((r, q)) => (r.to_string(), Some(q.to_ascii_uppercase())),
            None => (rest.to_string(), None),
        };
        let files = hf::list_files(&repo, "main").await?;
        let q = quant.unwrap_or_else(|| "Q4_K_M".into());
        let f = files
            .iter()
            .filter(|f| f.path.ends_with(".gguf") && !f.path.contains("mmproj"))
            .find(|f| spec::gguf_quant(&f.path).as_deref() == Some(q.as_str()))
            .with_context(|| format!("{repo} has no {q} GGUF"))?;
        let mm = files.iter().find(|f| f.path.contains("mmproj"));
        return Ok((f.size + mm.map(|m| m.size).unwrap_or(0), mm.is_some()));
    }
    let (path, tag) = match name.rsplit_once(':') {
        Some((p, t)) => (p.to_string(), t.to_string()),
        None => (name.to_string(), "latest".into()),
    };
    let path = if path.contains('/') {
        path
    } else {
        format!("library/{path}")
    };
    let registry = std::env::var("TENDRIL_OLLAMA_REGISTRY")
        .unwrap_or_else(|_| "https://registry.ollama.ai".into());
    let url = format!("{registry}/v2/{path}/manifests/{tag}");
    let resp = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?
        .get(&url)
        .header(
            reqwest::header::ACCEPT,
            "application/vnd.docker.distribution.manifest.v2+json",
        )
        .send()
        .await?;
    if !resp.status().is_success() {
        bail!("HTTP {}", resp.status());
    }
    let v: serde_json::Value = resp.json().await?;
    let layers = v["layers"]
        .as_array()
        .context("no layers in the manifest")?;
    let size = layers.iter().filter_map(|l| l["size"].as_u64()).sum();
    let vision = layers.iter().any(|l| {
        l["mediaType"]
            .as_str()
            .is_some_and(|m| m.contains("projector"))
    });
    Ok((size, vision))
}

/// "qwen2.5:7b" → ~4.7 GB (4-bit weights plus a little).
fn guess_size(name: &str) -> Option<u64> {
    let tag = name.rsplit_once(':')?.1.to_ascii_lowercase();
    let num: String = tag
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let b: f64 = num.parse().ok()?;
    let scale = match tag[num.len()..].chars().next()? {
        'b' => 1e9,
        'm' => 1e6,
        _ => return None,
    };
    Some((b * scale * 0.6) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guesses() {
        assert_eq!(guess_size("qwen2.5:7b"), Some(4_200_000_000));
        assert_eq!(guess_size("smollm:360m"), Some(216_000_000));
        assert_eq!(guess_size("llama3.2"), None);
    }
}
