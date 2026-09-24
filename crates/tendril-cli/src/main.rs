//! tendril — plan and run LLMs across the machines you already have.

mod bench;
mod bench_report;
mod catalog;
mod client;
mod common;
mod doctor;
mod fit;
mod inspect;
mod models;
mod node;
mod plan;
mod run;
mod serve;
mod ui;
mod verify;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "tendril",
    version,
    about = "Plan and run LLMs across the machines you already have",
    long_about = "Tendril inspects a model, looks at your machines and decides how to run it: \
                  on one machine, split across several, or not at all — and explains why.",
    after_help = "Examples:\n  \
        tendril run qwen2.5-0.5b                       chat with a small model on this machine\n  \
        tendril serve Qwen/Qwen2.5-7B-Instruct         serve it; other machines can join\n  \
        tendril join --token XXXX-XXXX-XXXX-XXXX      (on another machine) contribute it\n  \
        tendril serve llama-3.2-3b qwen2.5-1.5b       several models sharing the same machines\n  \
        tendril plan gemma-2-9b --node air=m4:16 --node mini=m5:16 --link thunderbolt\n  \
        tendril fit llama-3.1-70b --node studio=m2-ultra:192\n  \
        tendril doctor"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Chat with a model on this machine (downloads it if needed).
    Run(run::RunArgs),
    /// Serve a model across this machine and any that join; web chat + OpenAI API.
    Serve(serve::ServeArgs),
    /// Contribute this machine to a `tendril serve` cluster.
    Join(serve::JoinArgs),
    /// List Tendril clusters on the local network.
    Discover(serve::DiscoverArgs),
    /// Chat with a running server from the terminal.
    Chat(client::ChatArgs),
    /// Show a running server's machines, plan and traffic.
    Status(client::StatusArgs),
    /// Measure a server (or a model in-process): latency, throughput, where time goes.
    Bench(bench::BenchArgs),
    /// Download a model from HuggingFace.
    Pull(PullArgs),
    /// Check that a split pipeline computes exactly what one machine would.
    Verify(verify::VerifyArgs),
    /// Developer tools (hidden).
    #[command(hide = true, name = "dev-tiny-model")]
    DevTinyModel {
        dir: std::path::PathBuf,
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 64)]
        hidden: usize,
    },
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

#[derive(clap::Args)]
struct PullArgs {
    /// HuggingFace id or catalog name.
    model: String,
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
        Cmd::Run(a) => run::run(a),
        Cmd::Serve(a) => serve::serve(a),
        Cmd::Join(a) => serve::join(a),
        Cmd::Discover(a) => serve::discover_cmd(a),
        Cmd::Chat(a) => client::chat(a),
        Cmd::Status(a) => client::status(a),
        Cmd::Pull(a) => models::ensure_local(&a.model, true)
            .map(|(d, n)| println!("{} {n} is ready at {}", ui::ok_mark(), d.display())),
        Cmd::Verify(a) => verify::run(a),
        Cmd::Bench(a) => bench::run(a),
        Cmd::DevTinyModel {
            dir,
            layers,
            hidden,
        } => tendril_engine::testing::write_tiny_llama(
            &dir,
            layers,
            hidden,
            1,
            candle_core::DType::BF16,
        )
        .map(|_| {
            println!(
                "wrote a random {layers}-layer test model to {}",
                dir.display()
            )
        }),
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
