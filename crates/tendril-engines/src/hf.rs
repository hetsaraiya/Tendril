//! HuggingFace lookups engines need: a repo's files and sizes, and reading
//! small files (config.json, GGUF headers) without downloading weights.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tendril_core::model::source::{hf_endpoint, hf_token};

/// One file of a repo.
#[derive(Clone, Debug, PartialEq)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
}

/// `org/name[@revision]` split into (repo, revision).
pub fn split_repo(s: &str) -> (String, String) {
    let s = s
        .trim()
        .trim_start_matches("hf://")
        .trim_start_matches("https://huggingface.co/");
    match s.split_once('@') {
        Some((r, v)) => (r.trim_matches('/').to_string(), v.to_string()),
        None => (s.trim_matches('/').to_string(), "main".into()),
    }
}

/// Looks like `org/name` (and not a local path).
pub fn is_repo_id(s: &str) -> bool {
    let (r, _) = split_repo(s);
    let parts: Vec<&str> = r.split('/').collect();
    parts.len() == 2
        && parts
            .iter()
            .all(|p| !p.is_empty() && *p != "." && *p != "..")
        && !s.starts_with('/')
        && !s.starts_with('.')
        && !s.starts_with('~')
}

pub fn client() -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(t) = hf_token() {
        if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {t}")) {
            headers.insert(reqwest::header::AUTHORIZATION, v);
        }
    }
    Ok(reqwest::Client::builder()
        .user_agent(concat!("tendril/", env!("CARGO_PKG_VERSION")))
        .default_headers(headers)
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60))
        .build()?)
}

/// `<endpoint>/<repo>/resolve/<revision>/<file>`.
pub fn file_url(repo: &str, revision: &str, file: &str) -> String {
    format!("{}/{repo}/resolve/{revision}/{file}", hf_endpoint())
}

fn check(st: reqwest::StatusCode, what: &str) -> Result<()> {
    if st.is_success() {
        return Ok(());
    }
    match st.as_u16() {
        401 | 403 => bail!("{what}: access denied — the model may be gated: accept its license on huggingface.co and set HF_TOKEN"),
        404 => bail!("{what}: not found on HuggingFace"),
        _ => bail!("{what}: HTTP {st}"),
    }
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    size: u64,
}

/// Every file in a repo with its size.
pub async fn list_files(repo: &str, revision: &str) -> Result<Vec<RepoFile>> {
    let c = client()?;
    let mut url = format!(
        "{}/api/models/{repo}/tree/{revision}?recursive=true",
        hf_endpoint()
    );
    let mut out = Vec::new();
    for _ in 0..50 {
        let resp = c
            .get(&url)
            .send()
            .await
            .with_context(|| format!("list files of {repo}"))?;
        check(resp.status(), repo)?;
        let next = resp
            .headers()
            .get(reqwest::header::LINK)
            .and_then(|v| v.to_str().ok())
            .and_then(next_link);
        let entries: Vec<TreeEntry> = resp.json().await.context("unexpected file list")?;
        out.extend(
            entries
                .into_iter()
                .filter(|e| e.kind == "file")
                .map(|e| RepoFile {
                    size: e.lfs.map(|l| l.size).unwrap_or(e.size),
                    path: e.path,
                }),
        );
        match next {
            Some(n) => url = n,
            None => break,
        }
    }
    Ok(out)
}

fn next_link(h: &str) -> Option<String> {
    h.split(',').find_map(|part| {
        let (u, rel) = part.split_once(';')?;
        rel.contains("rel=\"next\"").then(|| {
            u.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        })
    })
}

/// A small file of a repo as JSON (config.json, model_index.json…).
pub async fn json(repo: &str, revision: &str, file: &str) -> Result<serde_json::Value> {
    let url = file_url(repo, revision, file);
    let resp = client()?
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    check(resp.status(), &format!("{repo}/{file}"))?;
    resp.json()
        .await
        .with_context(|| format!("{file} is not JSON"))
}

/// The first `len` bytes of a remote file.
pub async fn head_bytes(url: &str, len: u64) -> Result<Vec<u8>> {
    let resp = client()?
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes=0-{}", len - 1))
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    check(resp.status(), url)?;
    let full = resp.status() == reqwest::StatusCode::OK;
    let mut b = resp.bytes().await?.to_vec();
    if full {
        b.truncate(len as usize);
    }
    Ok(b)
}

/// Read a remote GGUF's header (growing the range until it parses).
pub async fn gguf_header(url: &str) -> Result<tendril_core::model::gguf::GgufHeader> {
    let mut want = 4u64 << 20;
    loop {
        let bytes = head_bytes(url, want).await?;
        let got = bytes.len() as u64;
        match tendril_core::model::gguf::read_header(&bytes[..], None) {
            Ok(h) => return Ok(h),
            Err(_) if got >= want && want < (256 << 20) => want *= 4,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ids() {
        assert!(is_repo_id("Qwen/Qwen2.5-7B-Instruct"));
        assert!(is_repo_id("Qwen/Qwen2.5-7B-Instruct@main"));
        assert!(!is_repo_id("/models/x.gguf"));
        assert!(!is_repo_id("./a/b"));
        assert!(!is_repo_id("llama3.2"));
        assert_eq!(
            split_repo("hf://a/b@v1"),
            ("a/b".to_string(), "v1".to_string())
        );
        assert_eq!(
            next_link("<https://x/api?cursor=2>; rel=\"next\""),
            Some("https://x/api?cursor=2".into())
        );
    }
}
