//! Finding model files locally and downloading them from HuggingFace.

use crate::ui::*;
use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tendril_core::model::source::{hf_cache_snapshot, hf_endpoint, http_client, resolve, ModelRef};
use tendril_core::units::Bytes;

pub fn models_dir() -> PathBuf {
    std::env::var("TENDRIL_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::cache_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("tendril")
                .join("models")
        })
}

fn complete(dir: &Path) -> bool {
    dir.join("config.json").exists()
        && dir.join("tokenizer.json").exists()
        && dir.join(".tendril-complete").exists()
}

fn usable_snapshot(dir: &Path) -> bool {
    dir.join("config.json").exists()
        && dir.join("tokenizer.json").exists()
        && std::fs::read_dir(dir)
            .map(|d| {
                d.flatten()
                    .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            })
            .unwrap_or(false)
}

/// Resolve a model name to a local directory, downloading if needed.
/// Returns (directory, display name).
pub fn ensure_local(name: &str, allow_download: bool) -> Result<(PathBuf, String)> {
    let r = resolve(name)?;
    let (repo, revision) = match &r {
        ModelRef::LocalDir(p) => {
            if !p.join("tokenizer.json").exists() {
                bail!("{} has no tokenizer.json — Tendril needs the HuggingFace tokenizer file next to the weights", p.display());
            }
            return Ok((p.clone(), display_name(p)));
        }
        ModelRef::LocalGguf(_) | ModelRef::HfGguf { .. } => bail!(
            "running GGUF files is not supported yet (planning them is). Use the original safetensors repo and add --quantize q4_k or q8_0 to get the same memory savings."
        ),
        ModelRef::HfRepo { repo, revision } => (repo.clone(), revision.clone()),
        ModelRef::Catalog(e) => (e.repo.to_string(), "main".to_string()),
    };
    if let Some(snap) = hf_cache_snapshot(&repo, &revision) {
        if usable_snapshot(&snap) {
            return Ok((snap, repo));
        }
    }
    let dir = models_dir().join(repo.replace('/', "--"));
    if complete(&dir) {
        return Ok((dir, repo));
    }
    if !allow_download {
        bail!("{repo} is not downloaded yet. Run: tendril pull {repo}");
    }
    pull(&repo, &revision, &dir)?;
    Ok((dir, repo))
}

fn display_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| p.display().to_string())
}

const WANTED: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "model.safetensors.index.json",
];

/// Download the files needed to run `repo` into `dir`, resuming partial files.
pub fn pull(repo: &str, revision: &str, dir: &Path) -> Result<()> {
    let client = http_client()?;
    let api = format!("{}/api/models/{repo}/revision/{revision}", hf_endpoint());
    let info: serde_json::Value = client
        .get(&api)
        .send()
        .with_context(|| format!("cannot reach HuggingFace for {repo} (offline? set HF_ENDPOINT for a mirror)"))?
        .error_for_status()
        .map_err(|e| {
            if e.status().is_some_and(|s| s.as_u16() == 401 || s.as_u16() == 403) {
                anyhow::anyhow!("{repo} is gated or private: accept its license on huggingface.co and set HF_TOKEN")
            } else if e.status().is_some_and(|s| s.as_u16() == 404) {
                anyhow::anyhow!("{repo} was not found on HuggingFace")
            } else {
                anyhow::anyhow!("{e}")
            }
        })?
        .json()?;
    let files: Vec<(String, u64)> = info
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|f| {
                    let name = f.get("rfilename")?.as_str()?.to_string();
                    let size = f.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
                    Some((name, size))
                })
                .collect()
        })
        .unwrap_or_default();
    let mut want: Vec<(String, u64)> = files
        .into_iter()
        .filter(|(n, _)| {
            WANTED.contains(&n.as_str()) || (n.ends_with(".safetensors") && !n.contains('/'))
        })
        .collect();
    if !want.iter().any(|(n, _)| n.ends_with(".safetensors")) {
        bail!("{repo} has no .safetensors weights at the top level (Tendril can't run .bin/.gguf-only repos yet)");
    }
    if !want.iter().any(|(n, _)| n == "tokenizer.json") {
        bail!("{repo} has no tokenizer.json; Tendril needs the fast tokenizer file");
    }
    // Sizes are optional in the listing; HEAD the big files when missing.
    for (n, size) in want.iter_mut() {
        if *size == 0 && n.ends_with(".safetensors") {
            if let Ok(r) = client
                .head(format!("{}/{repo}/resolve/{revision}/{n}", hf_endpoint()))
                .send()
            {
                *size = r
                    .headers()
                    .get("x-linked-size")
                    .or(r.headers().get("content-length"))
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            }
        }
    }
    let total: u64 = want.iter().map(|(_, s)| *s).sum();
    std::fs::create_dir_all(dir)?;
    println!(
        "{} Downloading {} ({}) to {}",
        cyan("↓"),
        bold(repo),
        Bytes(total),
        dim(dir.display())
    );
    let pb = ProgressBar::new(total.max(1));
    pb.set_style(
        ProgressStyle::with_template(
            "  {bar:36.cyan/blue} {bytes:>10}/{total_bytes} {bytes_per_sec:>12} eta {eta}  {msg}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    let big_client = reqwest::blocking::Client::builder()
        .user_agent(concat!("tendril/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        .timeout(None)
        .default_headers({
            let mut h = reqwest::header::HeaderMap::new();
            if let Some(t) = tendril_core::model::source::hf_token() {
                if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {t}")) {
                    h.insert(reqwest::header::AUTHORIZATION, v);
                }
            }
            h
        })
        .build()?;
    for (name, size) in &want {
        pb.set_message(name.clone());
        let dest = dir.join(name);
        if dest.exists() && (*size == 0 || std::fs::metadata(&dest)?.len() == *size) {
            pb.inc(*size);
            continue;
        }
        let part = dir.join(format!("{name}.part"));
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let mut req = big_client.get(format!(
            "{}/{repo}/resolve/{revision}/{name}",
            hf_endpoint()
        ));
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let mut resp = req
            .send()
            .with_context(|| format!("download {name}"))?
            .error_for_status()?;
        let resumed = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(resumed)
            .write(true)
            .truncate(!resumed)
            .open(&part)?;
        if resumed {
            pb.inc(have);
        }
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = resp.read(&mut buf)?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])?;
            pb.inc(n as u64);
        }
        f.flush()?;
        drop(f);
        std::fs::rename(&part, &dest)?;
    }
    pb.finish_and_clear();
    std::fs::write(dir.join(".tendril-complete"), repo)?;
    println!("{} Downloaded {repo}", ok_mark());
    Ok(())
}
