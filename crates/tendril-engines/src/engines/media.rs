//! Image generation (ComfyUI, stable-diffusion.cpp) and speech-to-text
//! (whisper.cpp): one model file, one machine.

use crate::detect::{find_program, first_version, python, python_package, run_capture};
use crate::engine::*;
use crate::util::*;
use crate::{hf, spec};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use base64::Engine as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// A model given as a single file: a path on the server, `org/repo/file`
/// on HuggingFace, or (`bare_ok`) a file name the engine already has.
struct FileModel {
    id: String,
    size: u64,
    file: Option<RemoteFile>,
    local_path: Option<String>,
}

async fn file_model(
    input: &str,
    exts: &[&str],
    engine_dir: Option<&str>,
    bare_ok: bool,
) -> Result<FileModel> {
    let m = input.trim();
    let has_ext = |s: &str| exts.iter().any(|e| s.to_ascii_lowercase().ends_with(e));
    let p = PathBuf::from(m);
    if p.is_file() {
        let size = std::fs::metadata(&p)?.len();
        let id = p
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| m.into());
        return Ok(FileModel {
            id,
            size,
            file: None,
            local_path: Some(m.into()),
        });
    }
    let (body, rev) = hf::split_repo(m);
    let parts: Vec<&str> = body.split('/').collect();
    if parts.len() >= 3 && has_ext(&body) {
        let repo = format!("{}/{}", parts[0], parts[1]);
        let file = parts[2..].join("/");
        let listing = hf::list_files(&repo, &rev).await?;
        let size = listing
            .iter()
            .find(|f| f.path == file)
            .map(|f| f.size)
            .with_context(|| format!("{repo} has no file {file}"))?;
        let base = file.rsplit('/').next().unwrap_or(&file).to_string();
        let name = match engine_dir {
            Some(_) => base.clone(),
            None => format!("{}/{}", repo_dir(&repo, &rev), file),
        };
        return Ok(FileModel {
            id: base,
            size,
            file: Some(RemoteFile {
                url: hf::file_url(&repo, &rev, &file),
                name,
                size,
                engine_dir: engine_dir.map(String::from),
            }),
            local_path: None,
        });
    }
    if bare_ok && has_ext(m) && !m.contains('/') {
        return Ok(FileModel {
            id: m.into(),
            size: 0,
            file: None,
            local_path: None,
        });
    }
    bail!(
        "give a model file: org/repo/file{} on HuggingFace{}",
        exts.first().unwrap_or(&""),
        if bare_ok {
            ", or the name of a file the engine already has"
        } else {
            ", or a path on this server"
        }
    )
}

fn file_resolved(
    engine: &str,
    task: Task,
    fm: FileModel,
    working: u64,
    mut notes: Vec<String>,
) -> ResolvedModel {
    if fm.size > 0 {
        notes.insert(0, format!("{} model file", gib(fm.size)));
    } else {
        notes.insert(0, "Size unknown: the file is already on the machine".into());
    }
    ResolvedModel {
        engine: engine.into(),
        served_name: fm.id.clone(),
        id: fm.id.clone(),
        task,
        vision: false,
        layered: false,
        spec: spec::opaque(&fm.id, engine, fm.size, working),
        files: fm.file.into_iter().collect(),
        local_path: fm.local_path,
        params: serde_json::json!({ "size": fm.size }),
        notes,
    }
}

fn model_arg(m: &ResolvedModel) -> String {
    m.local_path.clone().unwrap_or_else(|| "{file:0}".into())
}

async fn program_detection(
    engine: &str,
    info: &EngineInfo,
    host: &HostFacts,
    prog: Option<PathBuf>,
) -> Detection {
    let Some(p) = prog else {
        return not_installed(info, host);
    };
    let version = run_capture(
        &[p.display().to_string(), "--version".into()],
        Duration::from_secs(10),
    )
    .await
    .and_then(|o| first_version(&o));
    let (accelerators, devices) = host_accelerators(host);
    Detection {
        engine: engine.into(),
        ready: true,
        version,
        accelerators,
        devices,
        roles: vec!["serve".into()],
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------

pub struct ComfyUi;

const COMFY: &str = "comfyui";

fn comfy_dir() -> Option<PathBuf> {
    let home = dirs::home_dir();
    [
        std::env::var("TENDRIL_COMFYUI_DIR").ok().map(PathBuf::from),
        std::env::var("COMFYUI_PATH").ok().map(PathBuf::from),
        home.as_ref().map(|h| h.join("ComfyUI")),
        home.as_ref().map(|h| h.join("comfy/ComfyUI")),
        home.as_ref().map(|h| h.join("Documents/ComfyUI")),
    ]
    .into_iter()
    .flatten()
    .find(|d| d.join("main.py").is_file())
}

#[async_trait]
impl Engine for ComfyUi {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: COMFY.into(),
            name: "ComfyUI".into(),
            summary: "Image generation with Stable Diffusion / SDXL / Flux checkpoints (node workflows); one machine per model.".into(),
            homepage: "https://github.com/comfyanonymous/ComfyUI".into(),
            tasks: vec![Task::ImageGeneration],
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Image],
            multi_machine: MultiMachine::Single,
            security: Security::LocalOnly,
            platforms: "macOS, Linux, Windows (NVIDIA, Apple Silicon or CPU)".into(),
            model_hint: "A checkpoint on HuggingFace (org/repo/file.safetensors) or the name of one in ComfyUI/models/checkpoints".into(),
            example_model: "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors".into(),
            requirements: vec![
                Requirement::new(
                    "ComfyUI",
                    "a ComfyUI checkout in ~/ComfyUI (or TENDRIL_COMFYUI_DIR)",
                    &[("any", "git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI && pip install -r ~/ComfyUI/requirements.txt")],
                ),
                Requirement::new(
                    "torch",
                    "PyTorch for the Python that runs ComfyUI",
                    &[("any", "pip install torch torchvision torchaudio (see pytorch.org for your GPU)")],
                ),
            ],
        }
    }

    async fn locate(&self) -> Option<Located> {
        let dir = comfy_dir()?;
        let py = python()?;
        Some(Located {
            programs: vec![(
                "server".into(),
                vec![py.display().to_string(), "main.py".into()],
            )],
            dir: Some(dir),
            version: None,
        })
    }

    async fn detect(&self, host: &HostFacts) -> Detection {
        let info = self.info();
        let mut d = Detection {
            engine: COMFY.into(),
            ..Default::default()
        };
        let dir = comfy_dir();
        if dir.is_none() {
            d.missing.push(info.requirements[0].missing(&host.os));
        }
        match python() {
            Some(py) => match python_package(&py, "torch").await {
                Some(v) => d.notes.push(format!("torch {v}")),
                None => d.missing.push(info.requirements[1].missing(&host.os)),
            },
            None => d.missing.push(Missing {
                name: "python3".into(),
                why: "runs ComfyUI".into(),
                install: Some("install Python 3.10+".into()),
            }),
        }
        d.ready = d.missing.is_empty();
        if d.ready {
            let (a, n) = host_accelerators(host);
            d.accelerators = a;
            d.devices = n;
            d.roles = vec!["serve".into()];
            if let Some(dir) = dir {
                d.version = std::fs::read_to_string(dir.join("comfyui_version.py"))
                    .ok()
                    .and_then(|s| s.split('"').nth(1).map(String::from));
            }
        }
        d
    }

    async fn resolve(&self, req: &ModelRequest) -> Result<ResolvedModel> {
        let fm = file_model(
            &req.model,
            &[".safetensors", ".ckpt"],
            Some("models/checkpoints"),
            true,
        )
        .await?;
        if fm.local_path.is_some() {
            bail!("ComfyUI loads checkpoints from its models/checkpoints folder: give the file's name there, or a HuggingFace file");
        }
        let working = if fm.size > 4 << 30 {
            4u64 << 30
        } else {
            2 << 30
        };
        Ok(file_resolved(
            COMFY,
            Task::ImageGeneration,
            fm,
            working,
            vec![],
        ))
    }

    fn roles(&self, m: &ResolvedModel, p: &Placement, _machines: &[Machine]) -> Result<Vec<Role>> {
        let s = single("ComfyUI", p)?;
        let mut l = launch(
            COMFY,
            "server",
            &["--listen", "127.0.0.1", "--port", "{port}"],
        )
        .ready(Ready::Http {
            path: "/system_stats".into(),
        });
        l.cwd = Some("{dir}".into());
        l.files = m.files.clone();
        l.ready_timeout = 600;
        Ok(vec![role("server", &s.machine, true, l)])
    }

    fn api(&self) -> Api {
        Api::Native
    }

    /// OpenAI `images/generations` → a ComfyUI txt2img workflow (or a
    /// workflow passed as `workflow`), then the images back as base64.
    async fn images(
        &self,
        c: &reqwest::Client,
        base: &str,
        m: &ResolvedModel,
        req: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let prompt = req["prompt"].as_str().unwrap_or("").to_string();
        let workflow = match req.get("workflow") {
            Some(w) if w.is_object() => w.clone(),
            _ => {
                if prompt.is_empty() {
                    bail!("prompt is required");
                }
                let (w, h) = parse_size(req["size"].as_str().unwrap_or("1024x1024"))?;
                txt2img(
                    &m.id,
                    &prompt,
                    req["negative_prompt"].as_str().unwrap_or(""),
                    w,
                    h,
                    req["n"].as_u64().unwrap_or(1).clamp(1, 8),
                    req["steps"].as_u64().unwrap_or(20).clamp(1, 150),
                    req["cfg"].as_f64().unwrap_or(7.0),
                    req["seed"].as_u64().unwrap_or_else(rand_seed),
                )
            }
        };
        let r: serde_json::Value = c
            .post(format!("{base}/prompt"))
            .json(&serde_json::json!({ "prompt": workflow, "client_id": "tendril" }))
            .send()
            .await?
            .error_for_status()
            .context("ComfyUI rejected the workflow")?
            .json()
            .await?;
        let id = r["prompt_id"]
            .as_str()
            .context("ComfyUI returned no prompt id")?
            .to_string();
        let t0 = Instant::now();
        let outputs = loop {
            let h: serde_json::Value = c
                .get(format!("{base}/history/{id}"))
                .send()
                .await?
                .json()
                .await?;
            if let Some(entry) = h.get(&id) {
                if let Some(msg) = entry["status"]["messages"].as_array().and_then(|m| {
                    m.iter()
                        .find(|x| x[0] == "execution_error")
                        .map(|x| x[1]["exception_message"].clone())
                }) {
                    bail!(
                        "ComfyUI failed: {}",
                        msg.as_str().unwrap_or("execution error")
                    );
                }
                if entry["outputs"].as_object().is_some_and(|o| !o.is_empty()) {
                    break entry["outputs"].clone();
                }
            }
            if t0.elapsed() > Duration::from_secs(1800) {
                bail!("ComfyUI took too long");
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        };
        let mut data = Vec::new();
        for (_, node) in outputs.as_object().into_iter().flatten() {
            for img in node["images"].as_array().into_iter().flatten() {
                let bytes = c
                    .get(format!("{base}/view"))
                    .query(&[
                        ("filename", img["filename"].as_str().unwrap_or("")),
                        ("subfolder", img["subfolder"].as_str().unwrap_or("")),
                        ("type", img["type"].as_str().unwrap_or("output")),
                    ])
                    .send()
                    .await?
                    .error_for_status()?
                    .bytes()
                    .await?;
                data.push(serde_json::json!({
                    "b64_json": base64::engine::general_purpose::STANDARD.encode(&bytes),
                    "revised_prompt": prompt,
                }));
            }
        }
        if data.is_empty() {
            bail!("ComfyUI produced no images");
        }
        Ok(serde_json::json!({ "created": unix(), "data": data }))
    }

    fn transfer_note(&self, m: &ResolvedModel, _p: &Placement, _stage: usize) -> String {
        match m.files.first() {
            Some(f) => format!(
                "downloads {} ({}) into ComfyUI/models/checkpoints",
                f.name,
                gib(f.size)
            ),
            None => format!("uses {} from ComfyUI/models/checkpoints", m.id),
        }
    }
}

fn txt2img(
    ckpt: &str,
    prompt: &str,
    negative: &str,
    w: u64,
    h: u64,
    n: u64,
    steps: u64,
    cfg: f64,
    seed: u64,
) -> serde_json::Value {
    serde_json::json!({
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal",
            "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": n}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "tendril", "images": ["8", 0]}}
    })
}

fn parse_size(s: &str) -> Result<(u64, u64)> {
    let (w, h) = s.split_once('x').context("size must look like 1024x1024")?;
    let (w, h): (u64, u64) = (w.trim().parse()?, h.trim().parse()?);
    if !(64..=4096).contains(&w) || !(64..=4096).contains(&h) {
        bail!("size must be between 64 and 4096 pixels per side");
    }
    Ok((w / 8 * 8, h / 8 * 8))
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64 % 1_000_000_007)
        .unwrap_or(42)
}

fn unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------

pub struct SdCpp;

const SDCPP: &str = "sd.cpp";

#[async_trait]
impl Engine for SdCpp {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: SDCPP.into(),
            name: "stable-diffusion.cpp".into(),
            summary: "Image generation in plain C++ (SD 1.x/2.x, SDXL, Flux, GGUF) with an OpenAI-style images API; one machine per model.".into(),
            homepage: "https://github.com/leejet/stable-diffusion.cpp".into(),
            tasks: vec![Task::ImageGeneration],
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Image],
            multi_machine: MultiMachine::Single,
            security: Security::LocalOnly,
            platforms: "macOS, Linux, Windows".into(),
            model_hint: "A single-file checkpoint: org/repo/file.safetensors (or .gguf) on HuggingFace, or a path on this server".into(),
            example_model: "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors".into(),
            requirements: vec![Requirement::new(
                "sd-server",
                "serves the image model",
                &[("any", "download a release from https://github.com/leejet/stable-diffusion.cpp/releases (sd-server), or build it with cmake")],
            )],
        }
    }

    async fn locate(&self) -> Option<Located> {
        let p = find_program(&["sd-server"])?;
        Some(Located {
            programs: vec![("server".into(), vec![p.display().to_string()])],
            ..Default::default()
        })
    }

    async fn detect(&self, host: &HostFacts) -> Detection {
        program_detection(SDCPP, &self.info(), host, find_program(&["sd-server"])).await
    }

    async fn resolve(&self, req: &ModelRequest) -> Result<ResolvedModel> {
        let fm = file_model(&req.model, &[".safetensors", ".gguf", ".ckpt"], None, false).await?;
        let working = (fm.size / 3).max(2 << 30);
        Ok(file_resolved(
            SDCPP,
            Task::ImageGeneration,
            fm,
            working,
            vec![],
        ))
    }

    fn roles(&self, m: &ResolvedModel, p: &Placement, _machines: &[Machine]) -> Result<Vec<Role>> {
        let s = single("stable-diffusion.cpp", p)?;
        let mut l = launch(SDCPP, "server", &[])
            .args(["-m".to_string(), model_arg(m)])
            .args(["--listen-ip", "127.0.0.1", "--listen-port", "{port}"])
            .ready(Ready::Port);
        l.files = m.files.clone();
        l.ready_timeout = 600;
        Ok(vec![role("server", &s.machine, true, l)])
    }

    fn transfer_note(&self, m: &ResolvedModel, _p: &Placement, _stage: usize) -> String {
        match m.files.first() {
            Some(f) => format!("downloads {} ({})", m.id, gib(f.size)),
            None => "reads the model file on this server".into(),
        }
    }
}

// ---------------------------------------------------------------------------

pub struct WhisperCpp;

const WHISPER: &str = "whisper.cpp";

#[async_trait]
impl Engine for WhisperCpp {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: WHISPER.into(),
            name: "whisper.cpp".into(),
            summary: "Speech-to-text with OpenAI Whisper models in C++ (Metal, CUDA, CPU); one machine per model.".into(),
            homepage: "https://github.com/ggml-org/whisper.cpp".into(),
            tasks: vec![Task::Transcription],
            inputs: vec![Modality::Audio],
            outputs: vec![Modality::Text],
            multi_machine: MultiMachine::Single,
            security: Security::LocalOnly,
            platforms: "macOS, Linux, Windows".into(),
            model_hint: "A Whisper size (base.en, small, medium, large-v3, large-v3-turbo), org/repo/ggml-*.bin, or a path on this server".into(),
            example_model: "large-v3-turbo".into(),
            requirements: vec![Requirement::new(
                "whisper-server",
                "serves the speech model",
                &[
                    ("macos", "brew install whisper-cpp"),
                    ("any", "build https://github.com/ggml-org/whisper.cpp with cmake (target whisper-server)"),
                ],
            )],
        }
    }

    async fn locate(&self) -> Option<Located> {
        let p = find_program(&["whisper-server"])?;
        Some(Located {
            programs: vec![("server".into(), vec![p.display().to_string()])],
            ..Default::default()
        })
    }

    async fn detect(&self, host: &HostFacts) -> Detection {
        program_detection(
            WHISPER,
            &self.info(),
            host,
            find_program(&["whisper-server"]),
        )
        .await
    }

    async fn resolve(&self, req: &ModelRequest) -> Result<ResolvedModel> {
        let m = req.model.trim();
        let input = if !m.contains('/') && !m.contains('.')
            || matches!(m, "base.en" | "tiny.en" | "small.en" | "medium.en")
        {
            format!("ggerganov/whisper.cpp/ggml-{m}.bin")
        } else {
            m.to_string()
        };
        let fm = file_model(&input, &[".bin", ".gguf"], None, false).await?;
        let working = fm.size / 2 + (256 << 20);
        Ok(file_resolved(
            WHISPER,
            Task::Transcription,
            fm,
            working,
            vec![],
        ))
    }

    fn roles(&self, m: &ResolvedModel, p: &Placement, _machines: &[Machine]) -> Result<Vec<Role>> {
        let s = single("whisper.cpp", p)?;
        let mut l = launch(WHISPER, "server", &[])
            .args(["-m".to_string(), model_arg(m)])
            .args(["--host", "127.0.0.1", "--port", "{port}"])
            .ready(Ready::Port);
        l.files = m.files.clone();
        l.ready_timeout = 300;
        Ok(vec![role("server", &s.machine, true, l)])
    }

    fn api(&self) -> Api {
        Api::Native
    }

    async fn transcribe(
        &self,
        c: &reqwest::Client,
        base: &str,
        _m: &ResolvedModel,
        audio: Vec<u8>,
        filename: String,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let mut form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(audio).file_name(filename),
            )
            .text("response_format", "json");
        for k in ["language", "temperature", "prompt"] {
            if let Some(v) = params.get(k) {
                let s = v
                    .as_str()
                    .map(String::from)
                    .unwrap_or_else(|| v.to_string());
                form = form.text(k, s);
            }
        }
        let r = c
            .post(format!("{base}/inference"))
            .multipart(form)
            .send()
            .await?;
        let st = r.status();
        let body = r.text().await?;
        if !st.is_success() {
            bail!("whisper.cpp: HTTP {st}: {body}");
        }
        let v: serde_json::Value =
            serde_json::from_str(&body).unwrap_or_else(|_| serde_json::json!({ "text": body }));
        if let Some(e) = v.get("error") {
            bail!("whisper.cpp: {e}");
        }
        Ok(serde_json::json!({ "text": v["text"].as_str().unwrap_or("").trim() }))
    }

    fn transfer_note(&self, m: &ResolvedModel, _p: &Placement, _stage: usize) -> String {
        match m.files.first() {
            Some(f) => format!("downloads {} ({})", m.id, gib(f.size)),
            None => "reads the model file on this server".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comfy_workflow() {
        let w = txt2img("x.safetensors", "a cat", "", 512, 768, 2, 20, 7.0, 1);
        assert_eq!(w["5"]["inputs"]["height"], 768);
        assert_eq!(w["4"]["inputs"]["ckpt_name"], "x.safetensors");
        assert_eq!(parse_size("1000x600").unwrap(), (1000, 600));
        assert!(parse_size("10x10").is_err());
    }
}
