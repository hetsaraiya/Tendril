//! llama.cpp: `llama-server` serves the API; `ggml-rpc-server` on other
//! machines holds layers of a model too big for one.
//!
//! llama.cpp's RPC protocol has no authentication or encryption ("never run
//! the RPC server on an open network"), so every `ggml-rpc-server` listens on
//! 127.0.0.1 only and `llama-server` reaches it through Tendril's encrypted,
//! token-authenticated tunnels.

use crate::detect::{find_program, first_version, run_capture};
use crate::engine::*;
use crate::util::*;
use crate::{hf, spec};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tendril_core::model::gguf::{read_header, GgufHeader};

pub struct LlamaCpp;

pub const ID: &str = "llama.cpp";
const SERVER: &[&str] = &["llama-server"];
const RPC: &[&str] = &["ggml-rpc-server", "rpc-server", "llama-rpc-server"];

/// Quantizations tried, best first, when the user doesn't pick one.
const PREFERRED: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q4_K_L", "Q5_K_M", "Q4_0", "IQ4_XS", "Q6_K", "Q8_0", "Q5_K_S", "Q3_K_M",
    "Q3_K_L", "MXFP4", "BF16", "F16",
];

#[async_trait]
impl Engine for LlamaCpp {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: ID.into(),
            name: "llama.cpp".into(),
            summary: "GGUF models on any hardware (Mac, NVIDIA, AMD, CPU); splits a model's layers across mixed machines.".into(),
            homepage: "https://github.com/ggml-org/llama.cpp".into(),
            tasks: vec![Task::Chat],
            inputs: vec![Modality::Text, Modality::Image],
            outputs: vec![Modality::Text],
            multi_machine: MultiMachine::Layers,
            security: Security::TendrilTunnel {
                what: "llama.cpp's RPC traffic (layers and activations) is unencrypted and unauthenticated; Tendril binds every rpc-server to localhost and carries it over its own encrypted, token-authenticated links.".into(),
            },
            platforms: "macOS, Linux, Windows".into(),
            model_hint: "A GGUF repo (org/name, optionally :Q4_K_M), a .gguf file in a repo (org/name/file.gguf) or a .gguf path on this server".into(),
            example_model: "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M".into(),
            requirements: vec![
                Requirement::new(
                    "llama-server",
                    "serves the model's API",
                    &[
                        ("macos", "brew install llama.cpp"),
                        ("linux", "download a build from https://github.com/ggml-org/llama.cpp/releases (or: brew install llama.cpp)"),
                        ("windows", "winget install llama.cpp"),
                    ],
                ),
                Requirement::new(
                    "ggml-rpc-server",
                    "holds layers of a model another machine serves (only for models split across machines)",
                    &[(
                        "any",
                        "build llama.cpp with RPC: cmake -B build -DGGML_RPC=ON && cmake --build build --target ggml-rpc-server llama-server (add -DGGML_CUDA=ON on NVIDIA)",
                    )],
                ),
            ],
        }
    }

    async fn locate(&self) -> Option<Located> {
        let mut programs = Vec::new();
        if let Some(p) = find_program(SERVER) {
            programs.push(("server".to_string(), vec![p.display().to_string()]));
        }
        if let Some(p) = find_program(RPC) {
            programs.push(("rpc-server".to_string(), vec![p.display().to_string()]));
        }
        (!programs.is_empty()).then_some(Located {
            programs,
            ..Default::default()
        })
    }

    async fn detect(&self, host: &HostFacts) -> Detection {
        let info = self.info();
        let Some(loc) = self.locate().await else {
            return not_installed(&info, host);
        };
        let mut d = Detection {
            engine: ID.into(),
            ready: true,
            ..Default::default()
        };
        let probe = loc.program("server").or(loc.program("rpc-server")).cloned();
        if let Some(server) = loc.program("server") {
            d.roles.push("serve".into());
            let mut cmd = server.clone();
            cmd.push("--version".into());
            if let Some(out) = run_capture(&cmd, Duration::from_secs(15)).await {
                d.version = out
                    .lines()
                    .find(|l| l.starts_with("version"))
                    .map(|l| l.trim_start_matches("version:").trim().to_string())
                    .or_else(|| first_version(&out));
            }
        } else {
            d.missing.push(info.requirements[0].missing(&host.os));
        }
        if loc.program("rpc-server").is_some() {
            d.roles.push("split-worker".into());
        } else {
            d.missing.push(info.requirements[1].missing(&host.os));
        }
        if let Some(mut cmd) = probe {
            cmd.push("--list-devices".into());
            if let Some(out) = run_capture(&cmd, Duration::from_secs(20)).await {
                let devs = parse_devices(&out);
                d.devices = devs.len() as u32;
                for name in devs {
                    let a = accelerator(&name);
                    if !d.accelerators.contains(&a) {
                        d.accelerators.push(a);
                    }
                }
            }
        }
        if d.accelerators.is_empty() {
            d.accelerators.push("CPU".into());
        }
        if d.devices == 0 && (host.nvidia_gpus > 0 || host.apple_silicon) {
            d.notes.push("This llama.cpp build sees no GPU here, so it will run on the CPU. Install a build with GPU support (Metal on Mac, CUDA on NVIDIA).".into());
        }
        d
    }

    async fn resolve(&self, req: &ModelRequest) -> Result<ResolvedModel> {
        let (input, quant) = split_quant(&req.model, req.quantize.as_deref());
        let local = expand(&input);
        let (headers, files, mmproj, id, origin, local_path) = if local.exists() {
            let (parts, mmproj) = local_files(&local, quant.as_deref())?;
            let id = parts[0]
                .file_name()
                .map(|n| n.to_string_lossy().trim_end_matches(".gguf").to_string())
                .unwrap_or_else(|| input.clone());
            let first = parts[0].display().to_string();
            let mut headers = Vec::new();
            for p in &parts {
                let p = p.clone();
                headers.push(
                    tokio::task::spawn_blocking(move || -> Result<GgufHeader> {
                        let f = std::fs::File::open(&p)
                            .with_context(|| format!("open {}", p.display()))?;
                        let len = f.metadata().ok().map(|m| m.len());
                        read_header(std::io::BufReader::new(f), len)
                    })
                    .await??,
                );
            }
            let mm = mmproj.map(|p| {
                let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                (
                    RemoteFile {
                        url: String::new(),
                        name: p.display().to_string(),
                        size,
                        engine_dir: None,
                    },
                    size,
                )
            });
            (
                headers,
                vec![],
                mm,
                id,
                "local GGUF".to_string(),
                Some(first),
            )
        } else if hf::is_repo_id(&input) || input.ends_with(".gguf") {
            let ((headers, files), mm, id) = remote(&input, quant.as_deref()).await?;
            (headers, files, mm, id, "huggingface GGUF".to_string(), None)
        } else {
            bail!(
                "llama.cpp needs a GGUF model: an org/name repo with .gguf files, org/name/file.gguf, or a .gguf path on this server (got '{}')",
                req.model
            );
        };
        let mut header = headers[0].clone();
        for h in &headers[1..] {
            header.tensors.extend(h.tensors.iter().cloned());
        }
        let mut spec = header.to_spec(&id)?;
        spec.origin = origin;
        let mut files = files;
        let mut params = serde_json::json!({ "parts": files.len().max(1), "mmproj": null });
        let vision = mmproj.is_some();
        if let Some((f, size)) = mmproj {
            // The vision projector lives with the input embeddings on the head.
            spec.bytes.embed += tendril_core::Bytes(size);
            if local_path.is_some() {
                params["mmproj_path"] = f.name.clone().into();
            } else {
                params["mmproj"] = files.len().into();
                files.push(f);
            }
        }
        let arch = header
            .meta
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        Ok(ResolvedModel {
            engine: ID.into(),
            served_name: id.clone(),
            id,
            task: Task::Chat,
            vision,
            layered: true,
            notes: vec![format!(
                "{} layers ({arch}), {}",
                spec.num_layers,
                spec.repr.label()
            )],
            spec,
            files,
            local_path,
            params,
        })
    }

    fn local_path_spans(&self) -> bool {
        // ggml-rpc-server receives its layers from llama-server.
        true
    }

    fn arrange(
        &self,
        model: &ResolvedModel,
        mut p: Placement,
        machines: &[Machine],
    ) -> Result<Placement> {
        let find = |key: &str| machines.iter().find(|m| m.key == key);
        if p.stages.is_empty() {
            bail!("nothing to place");
        }
        let names = |k: &str| {
            find(k)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| k.to_string())
        };
        let heads: Vec<usize> = (0..p.stages.len())
            .filter(|&i| find(&p.stages[i].machine).is_some_and(|m| self.can_head(&m.detection)))
            .collect();
        let head = if model.local_path.is_some() {
            *heads
                .iter()
                .find(|&&i| find(&p.stages[i].machine).is_some_and(|m| m.local))
                .context("the model file is on this server, so this server must serve it — share this server's compute (it runs llama-server) or use a HuggingFace model")?
        } else {
            let prefer_local = heads
                .iter()
                .find(|&&i| find(&p.stages[i].machine).is_some_and(|m| m.local));
            match prefer_local {
                Some(&i) => i,
                None => *heads
                    .iter()
                    .max_by_key(|&&i| p.stages[i].weight_bytes)
                    .with_context(|| {
                        format!(
                            "none of {} has llama-server installed, so none can serve the model",
                            p.stages
                                .iter()
                                .map(|s| names(&s.machine))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })?,
            }
        };
        for (i, s) in p.stages.iter().enumerate() {
            if i != head && !find(&s.machine).is_some_and(|m| self.can_join(&m.detection)) {
                bail!(
                    "{} would hold layers for {}, but it has no ggml-rpc-server (llama.cpp built with -DGGML_RPC=ON)",
                    names(&s.machine),
                    names(&p.stages[head].machine)
                );
            }
        }
        let head_gpu = find(&p.stages[head].machine).is_some_and(|m| m.detection.devices > 0);
        let head_stage = p.stages.remove(head);
        // llama.cpp puts RPC devices first and the head's own GPUs last;
        // layers it doesn't offload stay on the head's CPU, at the front.
        if head_gpu {
            p.stages.push(head_stage);
            p.head = p.stages.len() - 1;
        } else {
            p.stages.insert(0, head_stage);
            p.head = 0;
        }
        let counts: Vec<u64> = p.stages.iter().map(|s| s.layers.1 - s.layers.0).collect();
        for (s, r) in p.stages.iter_mut().zip(ranges(&counts)) {
            s.layers = r;
        }
        Ok(p)
    }

    fn roles(
        &self,
        model: &ResolvedModel,
        p: &Placement,
        machines: &[Machine],
    ) -> Result<Vec<Role>> {
        let find = |key: &str| machines.iter().find(|m| m.key == key);
        let n = model.spec.num_layers;
        let head = &p.stages[p.head];
        let head_devices = find(&head.machine)
            .map(|m| m.detection.devices)
            .unwrap_or(0);
        let mut roles = Vec::new();
        let mut rpc_ids = Vec::new();
        // Offloaded layers per device, in llama.cpp's device order.
        let mut device_layers: Vec<u64> = Vec::new();
        for (i, s) in p.stages.iter().enumerate() {
            if i == p.head {
                continue;
            }
            let id = format!("rpc-{}", rpc_ids.len() + 1);
            let devs = find(&s.machine)
                .map(|m| m.detection.devices.max(1))
                .unwrap_or(1);
            device_layers.extend(even_split(s.layers.1 - s.layers.0, devs as usize));
            let l = launch(ID, "rpc-server", &["-H", "127.0.0.1", "-p", "{port}", "-c"])
                .env("LLAMA_CACHE", "{cache}")
                .ready(Ready::Port);
            let mut r = role(
                &id,
                &s.machine,
                false,
                Launch {
                    ready_timeout: 120,
                    ..l
                },
            );
            r.launch.extra = serde_json::json!({ "layers": [s.layers.0, s.layers.1] });
            roles.push(r);
            rpc_ids.push(id);
        }
        let head_layers = head.layers.1 - head.layers.0;
        let ngl = if head_devices > 0 {
            device_layers.extend(even_split(head_layers, head_devices as usize));
            n + 1
        } else {
            // The head's own layers stay on its CPU: the first `head_layers`.
            n + 1 - head_layers
        };
        let ctx = p.context.max(512) * p.concurrency.max(1);
        let model_arg = match &model.local_path {
            Some(path) => path.clone(),
            None => "{file:0}".into(),
        };
        let mut l = launch(ID, "server", &[])
            .args(["-m", &model_arg, "--host", "127.0.0.1", "--port", "{port}"])
            .args(["-c".to_string(), ctx.to_string()])
            .args(["-np".to_string(), p.concurrency.max(1).to_string()])
            .args(["-ngl".to_string(), ngl.to_string()])
            .args(["--alias", &model.served_name, "--jinja", "--no-webui"]);
        if let Some(mm) = model.params.get("mmproj").and_then(|v| v.as_u64()) {
            l = l.args(["--mmproj".to_string(), format!("{{file:{mm}}}")]);
        } else if let Some(mm) = model.params.get("mmproj_path").and_then(|v| v.as_str()) {
            l = l.args(["--mmproj", mm]);
        }
        if !rpc_ids.is_empty() {
            let peers: Vec<String> = rpc_ids.iter().map(|r| format!("{{peer:{r}}}")).collect();
            l = l.args(["--rpc".to_string(), peers.join(",")]);
            if device_layers.len() > 1 {
                let ts = tensor_split(&device_layers, ngl.min(n + 1));
                l = l.args([
                    "--tensor-split".to_string(),
                    ts.iter()
                        .map(|v| fmt_split(*v))
                        .collect::<Vec<_>>()
                        .join(","),
                ]);
            }
        }
        l.files = if model.local_path.is_some() {
            vec![]
        } else {
            model.files.clone()
        };
        l.peers = rpc_ids.clone();
        l.ready_timeout = 3600;
        let mut r = role("server", &head.machine, true, l);
        r.after = rpc_ids;
        roles.push(r);
        Ok(roles)
    }

    fn transfer_note(&self, model: &ResolvedModel, p: &Placement, stage: usize) -> String {
        let total: u64 = model.files.iter().map(|f| f.size).sum();
        if stage == p.head {
            if model.local_path.is_some() {
                "reads the model file on this server".into()
            } else if p.stages.len() > 1 {
                format!("downloads the GGUF ({}) and sends each machine its layers over Tendril's encrypted link", gib(total))
            } else {
                format!("downloads the GGUF ({})", gib(total))
            }
        } else {
            format!(
                "receives {} of layers from the serving machine over Tendril's encrypted link (cached here after the first load)",
                gib(p.stages[stage].weight_bytes)
            )
        }
    }
}

/// "org/name:Q4_K_M" → ("org/name", Some("Q4_K_M")); an explicit
/// `quantize` wins.
fn split_quant(model: &str, quantize: Option<&str>) -> (String, Option<String>) {
    let m = model.trim();
    // Only a known quantization tag after the colon: `C:\m.gguf` is a path.
    let (base, tag) = match m.rsplit_once(':') {
        Some((b, t)) if !b.is_empty() && is_quant_tag(t) => (b.to_string(), Some(t.to_string())),
        _ => (m.to_string(), None),
    };
    let q = quantize
        .filter(|q| !q.is_empty() && *q != "native")
        .map(String::from)
        .or(tag);
    (base, q.map(|q| q.to_ascii_uppercase()))
}

/// Q4_K_M, IQ3_XS, Q8_0, BF16, F16, F32, MXFP4 …
fn is_quant_tag(t: &str) -> bool {
    let u = t.to_ascii_uppercase();
    let body = u.strip_prefix("IQ").or_else(|| u.strip_prefix('Q'));
    match body {
        Some(b) => {
            b.chars().next().is_some_and(|c| c.is_ascii_digit())
                && b.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        None => matches!(u.as_str(), "BF16" | "F16" | "F32" | "MXFP4"),
    }
}

fn expand(s: &str) -> PathBuf {
    match (s.strip_prefix("~/"), dirs::home_dir()) {
        (Some(rest), Some(h)) => h.join(rest),
        _ => PathBuf::from(s),
    }
}

/// The parts of a local GGUF model (and a vision projector next to it).
fn local_files(p: &Path, quant: Option<&str>) -> Result<(Vec<PathBuf>, Option<PathBuf>)> {
    let (dir, chosen) = if p.is_dir() {
        let names: Vec<String> = std::fs::read_dir(p)?
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.ends_with(".gguf") && !n.to_ascii_lowercase().contains("mmproj"))
            .collect();
        let pick =
            choose(&names, quant).with_context(|| format!("no .gguf model in {}", p.display()))?;
        (p.to_path_buf(), pick)
    } else {
        let name = p
            .file_name()
            .context("not a file")?
            .to_string_lossy()
            .to_string();
        (
            p.parent().unwrap_or(Path::new(".")).to_path_buf(),
            vec![name],
        )
    };
    let mut parts: Vec<PathBuf> = chosen.iter().map(|n| dir.join(n)).collect();
    // A split model named by its first part: find the rest.
    if parts.len() == 1 {
        let name = chosen[0].clone();
        if let Some((1, total)) = spec::gguf_part(&name) {
            let prefix = &name[..name.rfind("-00001-of-").unwrap_or(0)];
            parts = (1..=total)
                .map(|i| dir.join(format!("{prefix}-{i:05}-of-{total:05}.gguf")))
                .collect();
            if let Some(missing) = parts.iter().find(|p| !p.exists()) {
                bail!("{} is missing (part of a split GGUF)", missing.display());
            }
        }
    }
    let mmproj = std::fs::read_dir(&dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .find(|f| {
            f.file_name()
                .is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains("mmproj"))
                && f.extension().is_some_and(|e| e == "gguf")
        });
    Ok((parts, mmproj))
}

/// Pick a model's GGUF file(s) from `names` (paths in a repo): the requested
/// quantization, else the first of [`PREFERRED`] present. Split models return
/// all parts in order.
fn choose(names: &[String], quant: Option<&str>) -> Option<Vec<String>> {
    let mut by_quant: Vec<(String, Vec<String>)> = Vec::new();
    for n in names {
        let q = spec::gguf_quant(n).unwrap_or_else(|| "?".into());
        match by_quant.iter_mut().find(|(k, _)| *k == q) {
            Some((_, v)) => v.push(n.clone()),
            None => by_quant.push((q, vec![n.clone()])),
        }
    }
    let pick = match quant {
        Some(q) => by_quant.iter().find(|(k, _)| k.eq_ignore_ascii_case(q))?,
        None => PREFERRED
            .iter()
            .find_map(|p| by_quant.iter().find(|(k, _)| k == p))
            .or(by_quant.first())?,
    };
    let mut files = pick.1.clone();
    files.sort();
    // Several variants with the same tag (e.g. two unrelated files): prefer
    // one split set, else the first file.
    if files.iter().any(|f| spec::gguf_part(f).is_some()) {
        let first = files
            .iter()
            .find(|f| spec::gguf_part(f).is_some_and(|(i, _)| i == 1))?
            .clone();
        let prefix = first[..first.rfind("-00001-of-")?].to_string();
        files.retain(|f| f.starts_with(&prefix) && spec::gguf_part(f).is_some());
    } else {
        files.truncate(1);
    }
    Some(files)
}

type Remote = (
    (Vec<GgufHeader>, Vec<RemoteFile>),
    Option<(RemoteFile, u64)>,
    String,
);

/// A model on HuggingFace: pick the files and read their headers.
async fn remote(input: &str, quant: Option<&str>) -> Result<Remote> {
    let (body, revision) = hf::split_repo(input);
    let parts: Vec<&str> = body.split('/').collect();
    if parts.len() < 2 {
        bail!("'{input}' is not a HuggingFace repo");
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    let listing = hf::list_files(&repo, &revision).await?;
    let size_of = |f: &str| {
        listing
            .iter()
            .find(|x| x.path == f)
            .map(|x| x.size)
            .unwrap_or(0)
    };
    let ggufs: Vec<String> = listing
        .iter()
        .map(|f| f.path.clone())
        .filter(|f| f.ends_with(".gguf") && !f.to_ascii_lowercase().contains("mmproj"))
        .collect();
    if ggufs.is_empty() {
        bail!("{repo} has no .gguf files — llama.cpp runs GGUF models (look for a repo ending in -GGUF)");
    }
    let chosen = if parts.len() > 2 {
        let file = parts[2..].join("/");
        if !ggufs.contains(&file) {
            bail!("{repo} has no file {file}");
        }
        match spec::gguf_part(&file) {
            Some(_) => choose(std::slice::from_ref(&file), None)
                .filter(|v| v.len() > 1)
                .unwrap_or_else(|| {
                    let prefix = &file[..file
                        .rfind("-of-")
                        .and_then(|i| file[..i].rfind('-'))
                        .unwrap_or(0)];
                    let mut v: Vec<String> = ggufs
                        .iter()
                        .filter(|g| g.starts_with(prefix))
                        .cloned()
                        .collect();
                    v.sort();
                    v
                }),
            None => vec![file],
        }
    } else {
        match choose(&ggufs, quant) {
            Some(c) => c,
            None => {
                let mut have: Vec<String> =
                    ggufs.iter().filter_map(|g| spec::gguf_quant(g)).collect();
                have.sort();
                have.dedup();
                bail!(
                    "{repo} has no {} file; it has: {}",
                    quant.unwrap_or("GGUF"),
                    have.join(", ")
                );
            }
        }
    };
    let dir = repo_dir(&repo, &revision);
    let mut headers = Vec::new();
    let mut files = Vec::new();
    for f in &chosen {
        let url = hf::file_url(&repo, &revision, f);
        headers.push(
            hf::gguf_header(&url)
                .await
                .with_context(|| format!("read {f}"))?,
        );
        files.push(RemoteFile {
            url,
            name: format!("{dir}/{f}"),
            size: size_of(f),
            engine_dir: None,
        });
    }
    let mut mm: Vec<&hf::RepoFile> = listing
        .iter()
        .filter(|f| f.path.ends_with(".gguf") && f.path.to_ascii_lowercase().contains("mmproj"))
        .collect();
    mm.sort_by_key(|f| {
        let l = f.path.to_ascii_lowercase();
        (
            if l.contains("f16") {
                0
            } else if l.contains("q8") {
                1
            } else {
                2
            },
            f.size,
        )
    });
    let mmproj = mm.first().map(|f| {
        (
            RemoteFile {
                url: hf::file_url(&repo, &revision, &f.path),
                name: format!("{dir}/{}", f.path),
                size: f.size,
                engine_dir: None,
            },
            f.size,
        )
    });
    let id = match chosen[0].rsplit('/').next() {
        Some(f) if parts.len() > 2 => f.trim_end_matches(".gguf").to_string(),
        _ => match spec::gguf_quant(&chosen[0]) {
            Some(q) => format!("{repo}:{q}"),
            None => repo.clone(),
        },
    };
    Ok(((headers, files), mmproj, id))
}

/// Device names from `--list-devices` ("  CUDA0: NVIDIA … (24210 MiB, …)").
fn parse_devices(out: &str) -> Vec<String> {
    let mut on = false;
    let mut devs = Vec::new();
    for l in out.lines() {
        if l.trim_start().starts_with("Available devices") {
            on = true;
            continue;
        }
        if on {
            let t = l.trim();
            if !l.starts_with(' ') || t.is_empty() {
                if !devs.is_empty() || !t.is_empty() {
                    break;
                }
                continue;
            }
            if let Some((name, _)) = t.split_once(':') {
                if !name.contains(' ') {
                    devs.push(name.to_string());
                }
            }
        }
    }
    devs
}

fn accelerator(dev: &str) -> String {
    let d = dev.to_ascii_uppercase();
    for (k, v) in [
        ("CUDA", "CUDA"),
        ("MTL", "Metal"),
        ("METAL", "Metal"),
        ("VULKAN", "Vulkan"),
        ("ROCM", "ROCm"),
        ("HIP", "ROCm"),
        ("SYCL", "SYCL"),
        ("CANN", "CANN"),
        ("OPENCL", "OpenCL"),
    ] {
        if d.starts_with(k) {
            return v.into();
        }
    }
    dev.trim_end_matches(char::is_numeric).to_string()
}

/// `--tensor-split` values that make llama.cpp put exactly `layers[k]`
/// offloaded positions on device k. `act` is the number of offloaded
/// positions including the output layer (which lands on the last device).
///
/// llama.cpp assigns position p to the first device whose cumulative,
/// normalised split exceeds p/act; placing each boundary half a position
/// early makes the assignment exact despite float rounding.
fn tensor_split(layers: &[u64], act: u64) -> Vec<f64> {
    let mut out = Vec::with_capacity(layers.len());
    let (mut prev, mut start) = (0.0f64, 0u64);
    for (k, &c) in layers.iter().enumerate() {
        let cum = if k + 1 == layers.len() {
            act as f64
        } else {
            ((start + c) as f64 - 0.5).max(prev)
        };
        out.push(cum - prev);
        prev = cum;
        start += c;
    }
    out
}

fn fmt_split(v: f64) -> String {
    let s = format!("{v:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's own assignment (src/llama-model.cpp), in f32.
    fn assign(ts: &[f64], n_layer: u64, ngl: u64) -> Vec<Option<usize>> {
        let mut splits: Vec<f32> = Vec::new();
        let mut sum = 0f32;
        for v in ts {
            sum += *v as f32;
            splits.push(sum);
        }
        for s in &mut splits {
            *s /= sum;
        }
        let start = (n_layer + 1).saturating_sub(ngl);
        let act = ngl.min(n_layer + 1);
        (0..=n_layer)
            .map(|il| {
                if il < start || il - start >= act {
                    return None;
                }
                let f = (il - start) as f32 / act as f32;
                Some(splits.iter().position(|&s| s > f).unwrap_or(splits.len()))
            })
            .collect()
    }

    #[test]
    fn tensor_split_is_exact() {
        // 32 layers: rpc 10, rpc 7 + 0 (two devices), head GPU 15 (+output).
        for (counts, n, head_cpu) in [
            (vec![10u64, 7, 0, 15], 32u64, 0u64),
            (vec![1, 1], 2, 0),
            (vec![20, 60], 80, 0),
            (vec![0, 5, 5], 10, 0),
            (vec![12, 12], 28, 4),
        ] {
            let ngl = n + 1 - head_cpu;
            let ts = tensor_split(&counts, ngl);
            assert!(ts.iter().all(|v| *v >= 0.0), "{ts:?}");
            let got = assign(&ts, n, ngl);
            let mut want = vec![None; head_cpu as usize];
            for (k, c) in counts.iter().enumerate() {
                want.extend(std::iter::repeat_n(Some(k), *c as usize));
            }
            want.push(Some(counts.len() - 1)); // output layer
            assert_eq!(got, want, "counts {counts:?} ts {ts:?}");
        }
    }

    #[test]
    fn quant_choice() {
        let names: Vec<String> = [
            "a-Q2_K.gguf",
            "a-Q4_K_M.gguf",
            "a-Q8_0.gguf",
            "big/a-Q8_0-00001-of-00002.gguf",
            "big/a-Q8_0-00002-of-00002.gguf",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(choose(&names, None).unwrap(), vec!["a-Q4_K_M.gguf"]);
        assert_eq!(choose(&names, Some("q2_k")).unwrap(), vec!["a-Q2_K.gguf"]);
        let split: Vec<String> = names[3..].to_vec();
        assert_eq!(choose(&split, Some("Q8_0")).unwrap().len(), 2);
        assert!(choose(&names, Some("Q5_K_M")).is_none());
        assert_eq!(
            split_quant("Qwen/X-GGUF:q4_k_m", None),
            ("Qwen/X-GGUF".into(), Some("Q4_K_M".into()))
        );
        assert_eq!(
            split_quant("Qwen/X-GGUF", Some("Q8_0")).1.as_deref(),
            Some("Q8_0")
        );
        // Windows paths keep their drive letter.
        assert_eq!(split_quant("C:\\m.gguf", None), ("C:\\m.gguf".into(), None));
        assert_eq!(split_quant("D:\\models\\x.gguf", None).1, None);
        assert_eq!(split_quant("a/b:bf16", None).1.as_deref(), Some("BF16"));
    }

    #[test]
    fn devices() {
        let out = "load_backend: loaded CPU\nAvailable devices:\n  CUDA0: NVIDIA RTX 4090 (24210 MiB, 23000 MiB free)\n  CUDA1: NVIDIA RTX 4090 (24210 MiB, 23000 MiB free)\n";
        assert_eq!(parse_devices(out), vec!["CUDA0", "CUDA1"]);
        assert_eq!(accelerator("MTL0"), "Metal");
        assert!(parse_devices("Available devices:\n  (none)\n").is_empty());
    }

    fn machine(key: &str, devices: u32, roles: &[&str], local: bool) -> Machine {
        Machine {
            key: key.into(),
            name: key.into(),
            address: "10.0.0.1".into(),
            local,
            detection: Detection {
                engine: ID.into(),
                ready: true,
                devices,
                roles: roles.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            },
        }
    }

    fn model(layers: u64) -> ResolvedModel {
        let mut spec = crate::spec::opaque("m", "t", 1 << 30, 0);
        spec.num_layers = layers;
        ResolvedModel {
            engine: ID.into(),
            id: "m".into(),
            served_name: "m".into(),
            task: Task::Chat,
            vision: false,
            spec,
            layered: true,
            files: vec![RemoteFile {
                url: "https://x/m.gguf".into(),
                name: "x/m.gguf".into(),
                size: 1 << 30,
                engine_dir: None,
            }],
            local_path: None,
            params: serde_json::json!({}),
            notes: vec![],
        }
    }

    fn stage(m: &str, a: u64, b: u64) -> Stage {
        Stage {
            machine: m.into(),
            layers: (a, b),
            weight_bytes: (b - a) << 20,
        }
    }

    #[test]
    fn gpu_head_goes_last_and_cpu_head_first() {
        let both = ["serve", "split-worker"];
        let ms = vec![
            machine("mac", 1, &both, true),
            machine("pc", 1, &both, false),
            machine("cpu", 0, &both, false),
        ];
        let p = Placement {
            stages: vec![stage("mac", 0, 10), stage("pc", 10, 32)],
            head: 0,
            total_layers: 32,
            context: 4096,
            concurrency: 2,
        };
        let a = LlamaCpp.arrange(&model(32), p, &ms).unwrap();
        assert_eq!(
            a.stages
                .iter()
                .map(|s| s.machine.as_str())
                .collect::<Vec<_>>(),
            ["pc", "mac"]
        );
        assert_eq!(
            (a.head, a.stages[0].layers, a.stages[1].layers),
            (1, (0, 22), (22, 32))
        );
        let roles = LlamaCpp.roles(&model(32), &a, &ms).unwrap();
        assert_eq!(roles.len(), 2);
        assert_eq!(roles[0].machine, "pc");
        let head = &roles[1];
        assert_eq!(head.after, vec!["rpc-1"]);
        let args = head.launch.args.join(" ");
        assert!(args.contains("--rpc {peer:rpc-1}"), "{args}");
        assert!(args.contains("-ngl 33"), "{args}");
        assert!(args.contains("--tensor-split 21.5,11.5"), "{args}");
        // A CPU-only head keeps its layers first, on its CPU.
        let ms2 = vec![
            machine("cpu", 0, &both, true),
            machine("pc", 1, &both, false),
        ];
        let p = Placement {
            stages: vec![stage("pc", 0, 20), stage("cpu", 20, 32)],
            head: 0,
            total_layers: 32,
            context: 4096,
            concurrency: 1,
        };
        let a = LlamaCpp.arrange(&model(32), p, &ms2).unwrap();
        assert_eq!(a.stages[0].machine, "cpu");
        assert_eq!(a.stages[0].layers, (0, 12));
        let roles = LlamaCpp.roles(&model(32), &a, &ms2).unwrap();
        let args = roles[1].launch.args.join(" ");
        assert!(args.contains("-ngl 21"), "{args}");
        assert!(!args.contains("--tensor-split"), "{args}");
    }

    #[test]
    fn workers_need_rpc_server() {
        let ms = vec![
            machine("a", 1, &["serve"], true),
            machine("b", 1, &["serve"], false),
        ];
        let p = Placement {
            stages: vec![stage("a", 0, 10), stage("b", 10, 32)],
            head: 0,
            total_layers: 32,
            context: 4096,
            concurrency: 1,
        };
        let e = LlamaCpp
            .arrange(&model(32), p, &ms)
            .unwrap_err()
            .to_string();
        assert!(e.contains("ggml-rpc-server"), "{e}");
    }
}
