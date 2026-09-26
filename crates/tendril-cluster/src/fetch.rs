//! Downloading exactly the tensors one stage needs from a remote checkpoint
//! (HuggingFace or a mirror) with HTTP range requests.

use crate::proto::{TensorEntry, WeightSource};
use crate::shard::ShardWriter;
use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::{Path, PathBuf};
use tendril_core::model::source::{download_client, hf_endpoint};

/// A client for `base_url`. The machine's own HuggingFace token is attached
/// only when the URL is its configured HuggingFace endpoint, so a coordinator
/// can never make it send the token elsewhere.
pub fn client_for(base_url: &str) -> Result<reqwest::blocking::Client> {
    if !(base_url.starts_with("https://") || base_url.starts_with("http://")) {
        bail!("refusing to download weights from {base_url}");
    }
    if base_url.starts_with(&format!("{}/", hf_endpoint())) {
        return download_client();
    }
    Ok(reqwest::blocking::Client::builder()
        .user_agent(concat!("tendril/", env!("CARGO_PKG_VERSION")))
        .timeout(None)
        .build()?)
}

/// Download a stage's tensors into a shard at `path`, one range request per
/// tensor. `progress(done, total)` is called as bytes arrive.
// ponytail: one request per tensor, whole-tensor retry; merge neighbouring
// ranges / resume mid-tensor if downloads of big stages prove flaky or slow.
pub fn download_shard(
    client: &reqwest::blocking::Client,
    source: &WeightSource,
    entries: &[TensorEntry],
    path: &Path,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf> {
    if entries.len() != source.files.len() {
        bail!("weight source doesn't match the tensor list");
    }
    let mut w = ShardWriter::create(path, entries)?;
    let total = w.total;
    let base = source.base_url.trim_end_matches('/');
    let mut buf = vec![0u8; 1 << 20];
    for (e, (file, offset)) in entries.iter().zip(&source.files) {
        if file.is_empty() || file.contains("..") || file.starts_with('/') {
            bail!("invalid checkpoint file name {file:?}");
        }
        if e.len == 0 {
            continue;
        }
        let url = format!("{base}/{file}");
        let before = w.written();
        let mut tries = 0;
        loop {
            let r: Result<()> = (|| {
                let end = offset.checked_add(e.len - 1).context("bad offset")?;
                let mut resp = client
                    .get(&url)
                    .header(reqwest::header::RANGE, format!("bytes={offset}-{end}"))
                    .send()
                    .with_context(|| format!("GET {url}"))?;
                match resp.status().as_u16() {
                    206 => {}
                    401 | 403 => bail!("{url}: access denied — the model may be gated: accept its license on huggingface.co and set HF_TOKEN on this machine"),
                    s => bail!("{url}: HTTP {s} (expected a byte range)"),
                }
                let mut at = 0u64;
                while at < e.len {
                    let want = buf.len().min((e.len - at) as usize);
                    let n = resp.read(&mut buf[..want])?;
                    if n == 0 {
                        bail!("connection closed early");
                    }
                    w.write(&e.name, at, &buf[..n])?;
                    at += n as u64;
                    progress(before + at, total);
                }
                Ok(())
            })();
            match r {
                Ok(()) => break,
                Err(err) if tries < 3 && !format!("{err}").contains("access denied") => {
                    tries += 1;
                    tracing::warn!("retrying {}: {err:#}", e.name);
                    w.rewind(before);
                }
                Err(err) => return Err(err),
            }
        }
    }
    w.finish()
}
