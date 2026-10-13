//! `tendril node`: this machine's profile as the planner sees it.

use crate::ui::{self, *};
use anyhow::Result;
use clap::Args;
use tendril_core::hardware::Backend;

#[derive(Args, Debug)]
pub struct NodeArgs {
    /// Measure this machine again (memory and CPU speed, a few seconds).
    /// It is otherwise measured once per hardware setup and reused.
    #[arg(long)]
    pub probe: bool,
    #[arg(long)]
    pub json: bool,
}

pub fn run(a: NodeArgs) -> Result<()> {
    if a.probe {
        let p = tendril_cluster::machine::this_machine(None, None);
        let mut b = tendril_core::hardware::measure(&p);
        b.measured_at = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();
        tendril_core::hardware::save_benchmark(&b);
    }
    let n = tendril_cluster::machine::this_machine(None, None);
    if a.json {
        println!("{}", serde_json::to_string_pretty(&n)?);
        return Ok(());
    }
    println!();
    println!("{} {}", bold("This machine ·"), bold(cyan(&n.name)));
    let mut rows = vec![
        (
            "Chip",
            format!(
                "{}{}",
                n.chip,
                n.gpu_cores
                    .map(|c| format!(" · {c} GPU cores"))
                    .unwrap_or_default()
            ),
        ),
        (
            "Compute",
            format!(
                "{} backend · {} CPU cores · ~{:.1} TFLOP/s fp16",
                n.backend.label(),
                n.cpu_cores,
                n.tflops
            ),
        ),
        (
            "Memory",
            format!(
                "{} total{}{}",
                n.total_memory,
                n.accel_memory
                    .map(|v| format!(" · {v} VRAM"))
                    .unwrap_or_default(),
                n.available_memory
                    .map(|v| format!(" · {v} free now"))
                    .unwrap_or_default()
            ),
        ),
        (
            "Model budget",
            format!(
                "{} {}",
                bold(n.usable_memory.to_string()),
                dim(format!("({})", n.usable_reason))
            ),
        ),
        (
            "Decode speed",
            match n.measured_bandwidth_gbs {
                Some(m) => format!(
                    "{} memory read measured · spec sheet {:.0} GB/s",
                    bold(format!("{m:.1} GB")),
                    n.bandwidth_gbs
                ),
                None => format!(
                    "{:.0} GB/s spec · {:.0} GB/s assumed {}",
                    n.bandwidth_gbs,
                    n.effective_bandwidth_gbs(),
                    dim("(not benchmarked yet)")
                ),
            },
        ),
        (
            "Prefill speed",
            match n.measured_tflops {
                Some(t) => format!("{} TFLOP/s measured", bold(format!("{t:.2}"))),
                None => format!("~{:.2} TFLOP/s assumed", n.effective_tflops()),
            },
        ),
        ("OS", format!("{} / {}", n.os, n.arch)),
    ];
    if let Some(b) = n.on_battery {
        rows.push((
            "Power",
            if b {
                yellow("on battery — expect throttling")
            } else {
                "plugged in".into()
            },
        ));
    }
    ui::kv(&rows);
    if n.backend == Backend::Metal && n.wired_limit.is_none() {
        println!();
        println!(
            "  {} {}",
            cyan("tip"),
            dim(format!(
                "macOS reserves ~1/3 of RAM from the GPU. For bigger models: sudo sysctl iogpu.wired_limit_mb={}",
                (n.total_memory.0 / (1 << 20)).saturating_sub(3584)
            ))
        );
    }
    if n.measured_bandwidth_gbs.is_none() {
        println!();
        println!(
            "{}",
            dim(if n.backend == Backend::Cpu {
                "Not measured yet: `tendril node --probe` measures it (serve and join do it automatically)."
            } else {
                "GPU speed uses spec-sheet numbers (each engine has its own kernels)."
            })
        );
    }
    Ok(())
}
