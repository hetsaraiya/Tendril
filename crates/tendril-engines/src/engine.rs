//! What every inference engine tells Tendril, and what Tendril asks of it.
//!
//! Tendril does not run models itself. It finds machines, measures them,
//! plans where a model goes, moves files, secures the network and serves one
//! API. An [`Engine`] is the part that actually runs the model: llama.cpp,
//! vLLM, Ollama, ComfyUI… Each engine is one type implementing this trait,
//! and the trait is where its behaviour lives: what it needs installed, what
//! it can run, whether it can span machines, how its traffic between
//! machines is secured, how to start it and how to talk to it.

use anyhow::{bail, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tendril_core::model::ModelSpec;

/// Kinds of input and output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
    Audio,
}

/// What a deployed model is for; decides the API and the chat mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    /// Chat / completions (text in, text out; images in for vision models).
    Chat,
    /// Prompt in, image out.
    ImageGeneration,
    /// Audio in, text out.
    Transcription,
}

/// Whether and how an engine runs one model across several machines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiMachine {
    /// One machine per model.
    Single,
    /// Layers split across machines of any kind (Mac, NVIDIA, CPU mixed).
    Layers,
    /// Split across machines that need the same GPUs (same type and count).
    Homogeneous,
    /// The engine forms its own cluster and decides the split itself.
    EngineManaged,
}

/// How an engine's traffic between machines is protected when a model spans
/// machines. (On one machine only the API leaves it, and Tendril always
/// carries that over its own encrypted link.)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Security {
    /// Never spans machines; only the API leaves the machine, through Tendril.
    LocalOnly,
    /// The engine talks plain TCP on ports Tendril knows, so Tendril binds it
    /// to localhost and carries it over its encrypted, authenticated links.
    TendrilTunnel { what: String },
    /// The engine encrypts and authenticates its own traffic.
    EngineEncrypted { what: String },
    /// The engine's own traffic is unencrypted and Tendril cannot wrap it
    /// (dynamic ports, its own discovery). Allowed only after the user
    /// confirms the machines share a trusted private network.
    TrustedNetworkOnly { what: String, risk: String },
}

impl Security {
    pub fn needs_trusted_network(&self) -> bool {
        matches!(self, Security::TrustedNetworkOnly { .. })
    }
}

/// Something a machine needs before it can run an engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Requirement {
    pub name: String,
    pub why: String,
    /// How to install it, per OS ("macos", "linux", "windows", "any").
    pub install: Vec<(String, String)>,
}

impl Requirement {
    pub fn new(name: &str, why: &str, install: &[(&str, &str)]) -> Requirement {
        Requirement {
            name: name.into(),
            why: why.into(),
            install: install
                .iter()
                .map(|(o, c)| (o.to_string(), c.to_string()))
                .collect(),
        }
    }

    /// The install hint for `os` (`std::env::consts::OS`), or a generic one.
    pub fn install_for(&self, os: &str) -> Option<String> {
        self.install
            .iter()
            .find(|(o, _)| o == os)
            .or_else(|| self.install.iter().find(|(o, _)| o == "any"))
            .map(|(_, c)| c.clone())
    }

    /// What this machine is missing, as reported to the dashboard.
    pub fn missing(&self, os: &str) -> Missing {
        Missing {
            name: self.name.clone(),
            why: self.why.clone(),
            install: self.install_for(os),
        }
    }
}

/// A requirement this machine doesn't meet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Missing {
    pub name: String,
    pub why: String,
    /// Command (or step) that installs it on this machine's OS.
    pub install: Option<String>,
}

/// Static facts about an engine (the dashboard's engine catalog).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineInfo {
    pub id: String,
    pub name: String,
    pub summary: String,
    pub homepage: String,
    pub tasks: Vec<Task>,
    pub inputs: Vec<Modality>,
    pub outputs: Vec<Modality>,
    pub multi_machine: MultiMachine,
    /// When a model spans machines.
    pub security: Security,
    /// Where it runs ("macOS, Linux, Windows").
    pub platforms: String,
    /// What to type as the model ("GGUF repo, e.g. Qwen/Qwen2.5-7B-Instruct-GGUF").
    pub model_hint: String,
    pub example_model: String,
    pub requirements: Vec<Requirement>,
}

/// What a machine found when it looked for an engine.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub engine: String,
    /// Installed and usable here.
    pub ready: bool,
    pub version: Option<String>,
    /// "CUDA", "Metal", "ROCm", "Vulkan", "CPU".
    pub accelerators: Vec<String>,
    /// Accelerator devices the engine can use here (0 = CPU only).
    pub devices: u32,
    /// What this machine can do with it ("serve" = run a model or the head
    /// of a split, "split-worker" = hold layers of a split model).
    pub roles: Vec<String>,
    pub missing: Vec<Missing>,
    pub notes: Vec<String>,
}

/// What a user asked to load.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRequest {
    pub model: String,
    /// Weight format / quantization preference (engine-specific meaning).
    #[serde(default)]
    pub quantize: Option<String>,
    pub context: usize,
    pub concurrency: usize,
}

/// A file a role downloads before it starts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemoteFile {
    pub url: String,
    /// File name in the machine's model cache (or under `dir`).
    pub name: String,
    pub size: u64,
    /// Put it under this directory of the engine's own install (e.g.
    /// ComfyUI's `models/checkpoints`) instead of Tendril's cache.
    #[serde(default)]
    pub engine_dir: Option<String>,
}

/// A model an engine can run, with everything needed to plan and start it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolvedModel {
    pub engine: String,
    /// What the user sees.
    pub id: String,
    /// The name the engine's API expects in requests.
    pub served_name: String,
    pub task: Task,
    /// Accepts images in chat.
    pub vision: bool,
    /// Memory model for planning. Models without layers that can be split
    /// (image, audio, opaque registry models) get a one-layer spec.
    pub spec: ModelSpec,
    /// `spec` has real layers the planner may split.
    pub layered: bool,
    /// Files the head role downloads.
    pub files: Vec<RemoteFile>,
    /// The model is a path on the server: its head must run there.
    pub local_path: Option<String>,
    /// Engine-specific facts for planning and launching.
    pub params: serde_json::Value,
    pub notes: Vec<String>,
}

/// A machine the pool offers to an engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Machine {
    /// Pool id ("0" is the server itself).
    pub key: String,
    pub name: String,
    /// LAN address the other machines reach it on.
    pub address: String,
    pub detection: Detection,
    pub local: bool,
}

/// One machine's part of a planned model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    pub machine: String,
    /// Transformer layers [start, end).
    pub layers: (u64, u64),
    pub weight_bytes: u64,
}

/// Where a model goes (from Tendril's planner), in the engine's stage order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Placement {
    pub stages: Vec<Stage>,
    /// The stage whose machine serves the API (and runs the engine's head).
    #[serde(default)]
    pub head: usize,
    pub total_layers: u64,
    pub context: usize,
    pub concurrency: usize,
}

/// When a started process is ready.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Ready {
    /// HTTP GET on this path answers 2xx.
    Http { path: String },
    /// The TCP port accepts connections.
    Port,
    /// The process has been running for a few seconds (workers that only
    /// connect out, like a vLLM headless node).
    Started,
}

/// How to start one engine process on one machine. Strings may contain
/// placeholders the agent fills in: `{port}` (a free local port the process
/// listens on), `{file:N}` (path of downloaded file N), `{peer:ROLE}`
/// (`127.0.0.1:P`, a local end of an encrypted tunnel to that role),
/// `{dir}` (the engine's install directory), `{namespace}` (a name derived
/// from the cluster token, for engines with their own discovery),
/// `{host}` (127.0.0.1, or 0.0.0.0 for `lan` processes), `{port:NAME}`
/// (another free port of this process, listed in `ports`),
/// `{peer_host:ROLE}` / `{peer_port:ROLE}` / `{peer_port:ROLE:NAME}` (the
/// LAN address of another role, for `lan` engines only), `{cache}` (a
/// cache directory for this engine on this machine).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Launch {
    pub engine: String,
    /// Which of the engine's programs (e.g. "server", "rpc-server").
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    /// Working directory (placeholders allowed).
    #[serde(default)]
    pub cwd: Option<String>,
    /// Extra ports the process needs besides `{port}` (`{port:NAME}`).
    #[serde(default)]
    pub ports: Vec<String>,
    pub files: Vec<RemoteFile>,
    /// Roles this process connects to through Tendril's tunnels.
    pub peers: Vec<String>,
    pub ready: Ready,
    /// Seconds to wait for readiness (model loading can be slow).
    pub ready_timeout: u64,
    /// Listens on the LAN (trusted-network engines) instead of 127.0.0.1.
    pub lan: bool,
    /// Engine-specific data for its `prepare` step on the machine.
    pub extra: serde_json::Value,
}

/// One process of a deployment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Role {
    pub id: String,
    pub machine: String,
    /// The API is served by this role.
    pub serves_api: bool,
    /// Roles that must be ready before this one starts.
    pub after: Vec<String>,
    /// Roles that must have started (not necessarily be ready) first.
    #[serde(default)]
    pub after_started: Vec<String>,
    pub launch: Launch,
}

/// How the API of a running engine is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Api {
    /// OpenAI-compatible: requests are forwarded as they are.
    OpenAi,
    /// The engine's own API; Tendril translates (images, transcription).
    Native,
}

/// What a located engine install looks like on a machine.
#[derive(Clone, Debug, Default)]
pub struct Located {
    /// Program name → command (program and leading arguments).
    pub programs: Vec<(String, Vec<String>)>,
    /// The engine's install directory, when it has one (`{dir}`).
    pub dir: Option<std::path::PathBuf>,
    pub version: Option<String>,
}

impl Located {
    pub fn program(&self, name: &str) -> Option<&Vec<String>> {
        self.programs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c)
    }
}

/// Facts about the machine an engine is detected on.
#[derive(Clone, Debug)]
pub struct HostFacts {
    pub os: String,
    pub arch: String,
    pub nvidia_gpus: u32,
    pub apple_silicon: bool,
}

impl HostFacts {
    pub fn this_machine() -> HostFacts {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        HostFacts {
            apple_silicon: os == "macos" && arch == "aarch64",
            nvidia_gpus: crate::detect::nvidia_gpus(),
            os,
            arch,
        }
    }
}

/// Reports what a role is doing while it prepares (downloads, pulls).
pub type Progress = std::sync::Arc<dyn Fn(&str, u64, u64) + Send + Sync>;

/// One inference engine.
#[async_trait]
pub trait Engine: Send + Sync {
    fn info(&self) -> EngineInfo;

    /// Find the engine's programs on this machine.
    async fn locate(&self) -> Option<Located>;

    /// Is it installed and usable here? What is missing? (Runs on each machine.)
    async fn detect(&self, host: &HostFacts) -> Detection;

    /// Turn what the user typed into a model this engine can run (config,
    /// files, memory). Runs on the server; may use the network.
    async fn resolve(&self, req: &ModelRequest) -> Result<ResolvedModel>;

    /// Can this machine run the part of the engine that serves the API?
    fn can_head(&self, d: &Detection) -> bool {
        d.ready && d.roles.iter().any(|r| r == "serve")
    }

    /// Can this machine hold part of a model another machine serves?
    fn can_join(&self, d: &Detection) -> bool {
        d.ready && d.roles.iter().any(|r| r == "split-worker")
    }

    /// Reorder or adjust Tendril's placement to how the engine actually
    /// lays layers out, so the plan shown is the plan that runs.
    fn arrange(
        &self,
        _model: &ResolvedModel,
        placement: Placement,
        _machines: &[Machine],
    ) -> Result<Placement> {
        Ok(placement)
    }

    /// The processes to start for `placement`.
    fn roles(
        &self,
        model: &ResolvedModel,
        placement: &Placement,
        machines: &[Machine],
    ) -> Result<Vec<Role>>;

    /// Engine-specific work on the machine after the process is ready
    /// (pull a model, create an instance). `base` is its local URL.
    async fn prepare(&self, _launch: &Launch, _base: &str, _progress: Progress) -> Result<()> {
        // Most engines are ready once their server answers.
        Ok(())
    }

    fn api(&self) -> Api {
        Api::OpenAi
    }

    /// `POST /v1/images/generations` for engines with their own image API.
    async fn images(
        &self,
        _client: &reqwest::Client,
        _base: &str,
        _model: &ResolvedModel,
        _req: serde_json::Value,
    ) -> Result<serde_json::Value> {
        bail!("{} doesn't generate images", self.info().name)
    }

    /// `POST /v1/audio/transcriptions` for engines with their own audio API.
    async fn transcribe(
        &self,
        _client: &reqwest::Client,
        _base: &str,
        _model: &ResolvedModel,
        _audio: Vec<u8>,
        _filename: String,
        _params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        bail!("{} doesn't transcribe audio", self.info().name)
    }

    /// What a stage moves over the network, for the plan preview.
    fn transfer_note(
        &self,
        _model: &ResolvedModel,
        _placement: &Placement,
        _stage: usize,
    ) -> String {
        "downloads the model".into()
    }
}
