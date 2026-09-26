//! `tendril node`: this machine's profile as the planner sees it.

use crate::ui::{self, *};
use anyhow::Result;
use clap::Args;
use tendril_core::hardware::{detect_local, probe_memory_bandwidth, Backend};

#[derive(Args, Debug)]
pub struct NodeArgs {
    /// Measure memory bandwidth (takes about a second).
    #[arg(long)]
    pub probe: bool,
    #[arg(long)]
    pub json: bool,
}

pub fn run(a: NodeArgs) -> Result<()> {
    let mut n = detect_local(true);
    if a.probe {
        let bw = probe_memory_bandwidth();
        n.measured_bandwidth_gbs = Some(bw);
    }
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
            "Bandwidth",
            match n.measured_bandwidth_gbs {
                Some(m) => format!(
                    "{m:.0} GB/s measured (copy) · spec {:.0} GB/s",
                    n.bandwidth_gbs
                ),
                None => format!(
                    "{:.0} GB/s spec · {:.0} GB/s used for estimates",
                    n.bandwidth_gbs,
                    n.effective_bandwidth_gbs()
                ),
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
    if !a.probe {
        println!();
        println!(
            "{}",
            dim("Run with --probe to measure memory bandwidth instead of using the spec sheet.")
        );
    }
    Ok(())
}
