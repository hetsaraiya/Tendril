//! tendril — plan and run LLMs across the machines you already have.

mod bench;
mod bench_report;
mod catalog;
mod client;
mod common;
mod doctor;
mod engines;
mod fit;
mod inspect;
mod node;
mod plan;
mod serve;
mod ui;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "tendril",
    version,
    about = "Run any model across the machines you already have, with the inference engine you choose",
    long_about = "Tendril finds your machines, plans where a model goes, starts the inference engine you \
                  pick (llama.cpp, vLLM, SGLang, Ollama, MLX, exo, ComfyUI, stable-diffusion.cpp, \
                  whisper.cpp) on them, encrypts the traffic between them and serves one \
                  OpenAI-compatible API.",
    after_help = "Examples:\n  \
        tendril engines                                which engines this machine has, and how to add one\n  \
        tendril serve                                  open a cluster; other machines can join\n  \
        tendril join --token XXXX-XXXX-XXXX-XXXX      (on another machine) contribute it\n  \
        tendril load Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --engine llama.cpp\n  \
        tendril load llama3.2:3b --engine ollama\n  \
        tendril load stabilityai/sdxl-turbo/sd_xl_turbo_1.0_fp16.safetensors --engine sd.cpp\n  \
        tendril chat\n  \
        tendril plan gemma-2-9b --node air=m4:16 --node mini=m5:16 --link thunderbolt\n  \
        tendril doctor"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Inference engines: which this machine has, and how to install the others.
    Engines(engines::EnginesArgs),
    /// Open a cluster on this machine; others join; dashboard + OpenAI API.
    Serve(serve::ServeArgs),
    /// Contribute this machine to a `tendril serve` cluster.
    Join(serve::JoinArgs),
    /// List Tendril clusters on the local network.
    Discover(serve::DiscoverArgs),
    /// Load a model on a running cluster with an engine (shows the plan first).
    Load(client::LoadArgs),
    /// Chat with a running server from the terminal.
    Chat(client::ChatArgs),
    /// Show a running server's machines, plan and traffic.
    Status(client::StatusArgs),
    /// Measure a running server: latency and throughput under load.
    Bench(bench::BenchArgs),
    /// Decide how to run a model on your machines, and explain why.
    Plan(plan::PlanArgs),
    /// "Can I run it?" matrix across quantizations and context lengths.
    Fit(fit::FitArgs),
    /// Show what a model contains without downloading its weights.
    Inspect(inspect::InspectArgs),
    /// Show this machine's hardware as the planner sees it.
    Node(node::NodeArgs),
    /// Check this machine's setup and suggest fixes.
    Doctor,
    /// List models Tendril knows offline.
    Models(catalog::ModelsArgs),
    /// List hardware presets usable with --node.
    Hardware,
}

fn main() {
    // Behave like a normal Unix tool when piped into `head`.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    if std::env::var("RUST_LOG").is_ok() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .try_init();
    }
    let r = match cli.cmd {
        Cmd::Engines(a) => engines::run(a),
        Cmd::Serve(a) => serve::serve(a),
        Cmd::Join(a) => serve::join(a),
        Cmd::Discover(a) => serve::discover_cmd(a),
        Cmd::Load(a) => client::load(a),
        Cmd::Chat(a) => client::chat(a),
        Cmd::Status(a) => client::status(a),
        Cmd::Bench(a) => bench::run(a),
        Cmd::Plan(a) => plan::run(a),
        Cmd::Fit(a) => fit::run(a),
        Cmd::Inspect(a) => inspect::run(a),
        Cmd::Node(a) => node::run(a),
        Cmd::Doctor => doctor::run(),
        Cmd::Models(a) => catalog::models(a),
        Cmd::Hardware => catalog::hardware(),
    };
    if let Err(e) = r {
        eprintln!("{} {}", ui::red("error:"), e);
        for cause in e.chain().skip(1) {
            eprintln!("  {} {}", ui::dim("caused by:"), cause);
        }
        std::process::exit(1);
    }
}
