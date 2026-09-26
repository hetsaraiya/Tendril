//! Resolving a user's model reference and inspecting it without downloading
//! weights: local directories, GGUF files, HuggingFace repos (headers only,
//! via HTTP range requests) and the built-in catalog.

use super::catalog::{self, CatalogEntry};
use super::config::spec_from_config;
use super::gguf;
use super::safetensors::{self, TensorInfo};
use super::ModelSpec;
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Debug)]
pub enum ModelRef {
    /// Directory with config.json and *.safetensors.
    LocalDir(PathBuf),
    /// A single .gguf file.
    LocalGguf(PathBuf),
    HfRepo {
        repo: String,
        revision: String,
    },
    HfGguf {
        repo: String,
        revision: String,
        file: String,
    },
    Catalog(&'static CatalogEntry),
}

impl ModelRef {
    pub fn describe(&self) -> String {
        match self {
            ModelRef::LocalDir(p) | ModelRef::LocalGguf(p) => p.display().to_string(),
            ModelRef::HfRepo { repo, revision } if revision == "main" => repo.clone(),
            ModelRef::HfRepo { repo, revision } => format!("{repo}@{revision}"),
            ModelRef::HfGguf { repo, file, .. } => format!("{repo}/{file}"),
            ModelRef::Catalog(e) => e.repo.to_string(),
        }
    }
}

pub fn hf_endpoint() -> String {
    std::env::var("TENDRIL_HF_ENDPOINT")
        .or_else(|_| std::env::var("HF_ENDPOINT"))
        .unwrap_or_else(|_| "https://huggingface.co".into())
        .trim_end_matches('/')
        .to_string()
}

pub fn hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        return Some(t);
    }
    let p = dirs::home_dir()?.join(".cache/huggingface/token");
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Interpret what the user typed.
pub fn resolve(input: &str) -> Result<ModelRef> {
    let s = input.trim();
    if s.is_empty() {
        bail!("empty model name");
    }
    let p = expand_home(s);
    if p.exists() {
        if p.is_dir() {
            if p.join("config.json").exists() {
                return Ok(ModelRef::LocalDir(p));
            }
            // A directory containing exactly one GGUF.
            let ggufs: Vec<PathBuf> = std::fs::read_dir(&p)?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
                .collect();
            if ggufs.len() == 1 {
                return Ok(ModelRef::LocalGguf(ggufs[0].clone()));
            }
            bail!(
                "{} has no config.json (and not exactly one .gguf file)",
                p.display()
            );
        }
        if p.extension().is_some_and(|e| e == "gguf") {
            return Ok(ModelRef::LocalGguf(p));
        }
        if p.file_name().is_some_and(|n| n == "config.json") {
            return Ok(ModelRef::LocalDir(
                p.parent().unwrap_or(Path::new(".")).to_path_buf(),
            ));
        }
        bail!("{} is not a model directory or .gguf file", p.display());
    }
    let s = s
        .strip_prefix("hf://")
        .or_else(|| s.strip_prefix("https://huggingface.co/"))
        .unwrap_or(s);
    let (body, revision) = match s.split_once('@') {
        Some((b, r)) => (b, r.to_string()),
        None => (s, "main".to_string()),
    };
    let parts: Vec<&str> = body.split('/').filter(|x| !x.is_empty()).collect();
    if parts.len() >= 3 && body.ends_with(".gguf") {
        return Ok(ModelRef::HfGguf {
            repo: format!("{}/{}", parts[0], parts[1]),
            revision,
            file: parts[2..].join("/"),
        });
    }
    if parts.len() == 2 {
        return Ok(ModelRef::HfRepo {
            repo: body.to_string(),
            revision,
        });
    }
    if let Some(e) = catalog::lookup(s) {
        return Ok(ModelRef::Catalog(e));
    }
    let hint = catalog::suggest(s);
    if hint.is_empty() {
        bail!("'{s}' is not a local path, a HuggingFace repo (org/name) or a known model. Run `tendril models` to see the catalog.")
    } else {
        bail!("unknown model '{s}'. Did you mean: {}?", hint.join(", "))
    }
}

fn expand_home(s: &str) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(h) = dirs::home_dir() {
            return h.join(rest);
        }
    }
    PathBuf::from(s)
}

/// Options that affect inspection.
#[derive(Clone, Debug, Default)]
pub struct InspectOptions {
    /// Never touch the network.
    pub offline: bool,
}

pub fn inspect(r: &ModelRef, opts: &InspectOptions) -> Result<ModelSpec> {
    match r {
        ModelRef::LocalDir(dir) => inspect_local_dir(dir),
        ModelRef::LocalGguf(p) => {
            let f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
            let len = f.metadata().ok().map(|m| m.len());
            let h = gguf::read_header(std::io::BufReader::new(f), len)?;
            h.to_spec(&p.display().to_string())
        }
        ModelRef::Catalog(e) => Ok(e.spec()),
        ModelRef::HfRepo { repo, revision } => {
            if let Some(dir) = hf_cache_snapshot(repo, revision) {
                let mut s = inspect_local_dir(&dir)?;
                s.id = repo.clone();
                s.origin = "huggingface (local cache)".into();
                return Ok(s);
            }
            if let Some(s) = read_meta_cache(repo, revision) {
                return Ok(s);
            }
            if opts.offline {
                return catalog_fallback(repo, "offline mode");
            }
            match inspect_hf_repo(repo, revision) {
                Ok(s) => {
                    write_meta_cache(repo, revision, &s);
                    Ok(s)
                }
                Err(e) => catalog_fallback(repo, &format!("{e:#}")).map_err(|_| e),
            }
        }
        ModelRef::HfGguf {
            repo,
            revision,
            file,
        } => {
            if opts.offline {
                bail!("cannot inspect a remote GGUF in offline mode");
            }
            let url = format!("{}/{repo}/resolve/{revision}/{file}", hf_endpoint());
            let client = http_client()?;
            let mut want = 4u64 << 20;
            loop {
                let bytes = fetch_range(&client, &url, 0, want)?;
                let got = bytes.len() as u64;
                match gguf::read_header(&bytes[..], None) {
                    Ok(h) => {
                        let mut s = h.to_spec(&format!("{repo}/{file}"))?;
                        s.origin = "huggingface gguf (header only)".into();
                        return Ok(s);
                    }
                    Err(e) if got >= want && want < (256 << 20) => {
                        let _ = e;
                        want *= 4;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }
}

fn catalog_fallback(repo: &str, why: &str) -> Result<ModelSpec> {
    let e = catalog::lookup(repo).ok_or_else(|| anyhow!("{why}"))?;
    let mut s = e.spec();
    s.notes.push(format!(
        "Using built-in catalog numbers because the HuggingFace lookup failed ({why})."
    ));
    Ok(s)
}

pub fn inspect_local_dir(dir: &Path) -> Result<ModelSpec> {
    let cfg_path = dir.join("config.json");
    let raw: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&cfg_path).with_context(|| format!("read {}", cfg_path.display()))?,
    )
    .context("config.json is not valid JSON")?;
    let mut spec = spec_from_config(&dir.display().to_string(), "local safetensors", &raw)?;
    let mut tensors: Vec<TensorInfo> = Vec::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    for f in &files {
        let (t, _) = safetensors::read_local_header(f)?;
        tensors.extend(t);
    }
    if tensors.is_empty() {
        spec.notes
            .push("No .safetensors files found; sizes are estimated from config.json.".into());
    } else {
        apply_measured(&mut spec, &tensors);
    }
    Ok(spec)
}

/// Replace formula byte counts with measured tensor sizes.
pub fn apply_measured(spec: &mut ModelSpec, tensors: &[TensorInfo]) {
    let c = safetensors::component_bytes(tensors, spec.num_layers as usize);
    if c.lm_head.0 == 0 && !spec.tie_embeddings {
        spec.tie_embeddings = true;
        spec.params.lm_head = 0;
    }
    if c.lm_head.0 > 0 && spec.tie_embeddings {
        spec.tie_embeddings = false;
    }
    // Derive the stored dtype from the dominant tensor dtype.
    let mut by_dtype: std::collections::BTreeMap<&str, u64> = Default::default();
    for t in tensors {
        *by_dtype.entry(t.dtype.as_str()).or_default() += t.len();
    }
    if let Some((dt, _)) = by_dtype.iter().max_by_key(|(_, b)| **b) {
        let q = match *dt {
            "F32" => Some(super::Quant::F32),
            "F16" => Some(super::Quant::F16),
            "BF16" => Some(super::Quant::Bf16),
            _ => None, // packed quantized (U32) keeps the config-derived label
        };
        if let Some(q) = q {
            spec.stored = q;
            spec.repr = q;
        }
    }
    spec.bytes = c;
    spec.bytes_measured = true;
}

/// Look for a model already downloaded by huggingface_hub or by Tendril.
pub fn hf_cache_snapshot(repo: &str, revision: &str) -> Option<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(h) = std::env::var("HF_HUB_CACHE") {
        roots.push(PathBuf::from(h));
    }
    if let Ok(h) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(h).join("hub"));
    }
    if let Some(h) = dirs::home_dir() {
        roots.push(h.join(".cache/huggingface/hub"));
    }
    let folder = format!("models--{}", repo.replace('/', "--"));
    for root in roots {
        let base = root.join(&folder);
        let rev = std::fs::read_to_string(base.join("refs").join(revision))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| revision.to_string());
        let snap = base.join("snapshots").join(&rev);
        if snap.join("config.json").exists() {
            return Some(snap);
        }
    }
    None
}

pub fn http_client() -> Result<reqwest::blocking::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(t) = hf_token() {
        if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {t}")) {
            headers.insert(reqwest::header::AUTHORIZATION, v);
        }
    }
    Ok(reqwest::blocking::Client::builder()
        .user_agent(concat!("tendril/", env!("CARGO_PKG_VERSION")))
        .default_headers(headers)
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .build()?)
}

fn check(resp: reqwest::blocking::Response, what: &str) -> Result<reqwest::blocking::Response> {
    let st = resp.status();
    if st.is_success() {
        return Ok(resp);
    }
    match st.as_u16() {
        401 | 403 => bail!(
            "{what}: access denied ({st}). This model may be gated: accept its license on huggingface.co and set HF_TOKEN."
        ),
        404 => bail!("{what}: not found (404)"),
        _ => bail!("{what}: HTTP {st}"),
    }
}

pub fn fetch_range(
    client: &reqwest::blocking::Client,
    url: &str,
    start: u64,
    len: u64,
) -> Result<Vec<u8>> {
    let resp = client
        .get(url)
        .header(
            reqwest::header::RANGE,
            format!("bytes={}-{}", start, start + len - 1),
        )
        .send()
        .with_context(|| format!("GET {url}"))?;
    let resp = check(resp, url)?;
    let full = resp.status() == reqwest::StatusCode::OK;
    let mut body = resp.bytes()?.to_vec();
    if full {
        // Server ignored the range; slice it ourselves.
        let s = (start as usize).min(body.len());
        let e = ((start + len) as usize).min(body.len());
        body = body[s..e].to_vec();
    }
    Ok(body)
}

/// Base URL of a repo's files: `<endpoint>/<repo>/resolve/<revision>`.
pub fn hf_resolve_base(repo: &str, revision: &str) -> String {
    format!("{}/{repo}/resolve/{revision}", hf_endpoint())
}

/// A client for large downloads: no overall timeout, only a connect timeout.
pub fn download_client() -> Result<reqwest::blocking::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(t) = hf_token() {
        if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {t}")) {
            headers.insert(reqwest::header::AUTHORIZATION, v);
        }
    }
    Ok(reqwest::blocking::Client::builder()
        .user_agent(concat!("tendril/", env!("CARGO_PKG_VERSION")))
        .default_headers(headers)
        .connect_timeout(Duration::from_secs(15))
        .timeout(None)
        .build()?)
}

/// One tensor of a remote checkpoint and where its bytes live.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RemoteTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Checkpoint file inside the repo, e.g. `model-00001-of-00004.safetensors`.
    pub file: String,
    /// Absolute byte offset of the tensor's data in `file`.
    pub offset: u64,
    pub len: u64,
}

impl RemoteTensor {
    /// The header entry in the form the inspector works with.
    pub fn info(&self) -> TensorInfo {
        TensorInfo {
            name: self.name.clone(),
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            start: 0,
            end: self.len,
        }
    }
}

/// The checkpoint files of a repo: from the shard index, else `model.safetensors`.
fn hf_checkpoint_files(client: &reqwest::blocking::Client, base: &str) -> Result<Vec<String>> {
    Ok(
        match client
            .get(format!("{base}/model.safetensors.index.json"))
            .send()
        {
            Ok(r) if r.status().is_success() => {
                let idx: serde_json::Value = r.json()?;
                let mut s: Vec<String> = idx
                    .get("weight_map")
                    .and_then(|w| w.as_object())
                    .map(|m| {
                        m.values()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                s.sort();
                s.dedup();
                s
            }
            _ => vec!["model.safetensors".to_string()],
        },
    )
}

/// Every tensor of a HuggingFace safetensors checkpoint with its location,
/// read from the file headers alone (two small range requests per file).
pub fn hf_tensor_index(repo: &str, revision: &str) -> Result<Vec<RemoteTensor>> {
    let client = http_client()?;
    let base = hf_resolve_base(repo, revision);
    let mut out = Vec::new();
    for file in hf_checkpoint_files(&client, &base)? {
        out.extend(remote_header(&client, &base, &file)?);
    }
    if out.is_empty() {
        bail!("{repo} has no safetensors weights");
    }
    Ok(out)
}

fn remote_header(
    client: &reqwest::blocking::Client,
    base: &str,
    file: &str,
) -> Result<Vec<RemoteTensor>> {
    if file.contains("..") || file.starts_with('/') {
        bail!("{file}: invalid checkpoint file name");
    }
    let url = format!("{base}/{file}");
    let head = fetch_range(client, &url, 0, 8)?;
    if head.len() != 8 {
        bail!("{file}: short read");
    }
    let n = u64::from_le_bytes(head.try_into().unwrap());
    if n > safetensors::MAX_HEADER {
        bail!("{file}: implausible header size {n}");
    }
    let json = fetch_range(client, &url, 8, n)?;
    Ok(safetensors::parse_header(&json)?
        .into_iter()
        .map(|t| RemoteTensor {
            offset: 8 + n + t.start,
            len: t.len(),
            name: t.name,
            dtype: t.dtype,
            shape: t.shape,
            file: file.to_string(),
        })
        .collect())
}

fn inspect_hf_repo(repo: &str, revision: &str) -> Result<ModelSpec> {
    let client = http_client()?;
    let base = hf_resolve_base(repo, revision);
    let cfg: serde_json::Value = check(
        client
            .get(format!("{base}/config.json"))
            .send()
            .with_context(|| format!("fetch {repo}/config.json"))?,
        &format!("{repo}/config.json"),
    )?
    .json()
    .context("config.json is not valid JSON")?;
    let mut spec = spec_from_config(repo, "huggingface (headers only)", &cfg)?;
    let files = hf_checkpoint_files(&client, &base)?;
    let mut tensors = Vec::new();
    for file in &files {
        match remote_header(&client, &base, file) {
            Ok(t) => tensors.extend(t.iter().map(RemoteTensor::info)),
            Err(_) if files.len() == 1 => {
                spec.notes.push("No safetensors weights found in the repo; sizes are estimated from config.json.".into());
                return Ok(spec);
            }
            Err(e) => return Err(e),
        }
    }
    apply_measured(&mut spec, &tensors);
    Ok(spec)
}

fn meta_cache_path(repo: &str, revision: &str) -> Option<PathBuf> {
    let d = dirs::cache_dir()?.join("tendril").join("meta");
    Some(d.join(format!("{}@{}.json", repo.replace('/', "--"), revision)))
}

fn read_meta_cache(repo: &str, revision: &str) -> Option<ModelSpec> {
    if revision == "main" {
        // "main" moves; re-check at most daily.
        let p = meta_cache_path(repo, revision)?;
        let age = std::fs::metadata(&p)
            .ok()?
            .modified()
            .ok()?
            .elapsed()
            .ok()?;
        if age > Duration::from_secs(86400) {
            return None;
        }
    }
    let p = meta_cache_path(repo, revision)?;
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

fn write_meta_cache(repo: &str, revision: &str, s: &ModelSpec) {
    if let Some(p) = meta_cache_path(repo, revision) {
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Ok(j) = serde_json::to_vec_pretty(s) {
            let _ = std::fs::write(p, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_forms() {
        assert!(matches!(
            resolve("Qwen/Qwen2.5-7B-Instruct").unwrap(),
            ModelRef::HfRepo { .. }
        ));
        match resolve("hf://org/repo@v2").unwrap() {
            ModelRef::HfRepo { repo, revision } => {
                assert_eq!(repo, "org/repo");
                assert_eq!(revision, "v2");
            }
            _ => panic!(),
        }
        assert!(matches!(
            resolve("bartowski/Foo-GGUF/Foo-Q4_K_M.gguf").unwrap(),
            ModelRef::HfGguf { .. }
        ));
        assert!(matches!(
            resolve("llama-3.1-8b").unwrap(),
            ModelRef::Catalog(_)
        ));
        let err = resolve("lama-3.1-8b").unwrap_err().to_string();
        assert!(err.contains("Did you mean"), "{err}");
    }

    #[test]
    fn local_dir_with_headers() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"llama","num_hidden_layers":2,"hidden_size":8,"intermediate_size":16,
                "num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        let header = br#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[10,8],"data_offsets":[0,320]},
            "model.layers.0.mlp.up_proj.weight":{"dtype":"F32","shape":[16,8],"data_offsets":[320,832]},
            "model.layers.1.mlp.up_proj.weight":{"dtype":"F32","shape":[16,8],"data_offsets":[832,1344]},
            "model.norm.weight":{"dtype":"F32","shape":[8],"data_offsets":[1344,1376]}}"#;
        let mut f = Vec::new();
        f.extend_from_slice(&(header.len() as u64).to_le_bytes());
        f.extend_from_slice(header);
        f.extend(std::iter::repeat_n(0u8, 1376));
        std::fs::write(dir.path().join("model.safetensors"), f).unwrap();
        let s = inspect(
            &resolve(dir.path().to_str().unwrap()).unwrap(),
            &Default::default(),
        )
        .unwrap();
        assert!(s.bytes_measured);
        assert_eq!(s.bytes.embed.0, 320);
        assert_eq!(s.bytes.layers[1].0, 512);
        assert_eq!(s.stored, super::super::Quant::F32);
    }
}
