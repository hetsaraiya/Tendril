//! tendril — plan and run LLMs across the machines you already have.

mod catalog;
mod common;
mod doctor;
mod fit;
mod inspect;
mod node;
mod plan;
mod ui;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "tendril",
    version,
    about = "Plan and run LLMs across the machines you already have",
    long_about = "Tendril inspects a model, looks at your machines and decides how to run it: \
                  on one machine, split across several, or not at all — and explains why.",
    after_help = "Examples:\n  \
        tendril plan gemma-2-9b --node air=m4:16 --node mini=m5:16 --link thunderbolt\n  \
        tendril plan Qwen/Qwen2.5-14B-Instruct --context 32k --explain\n  \
        tendril fit llama-3.1-70b --node studio=m2-ultra:192\n  \
        tendril inspect ./models/my-model\n  \
        tendril doctor"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
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
    let r = match cli.cmd {
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
