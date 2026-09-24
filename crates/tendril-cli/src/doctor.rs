//! `tendril doctor`: is this machine ready, and what would help?

use crate::ui::*;
use anyhow::Result;
use std::time::Duration;
use tendril_core::hardware::{detect_local, Backend};
use tendril_core::model::source::{hf_endpoint, hf_token};
use tendril_core::units::{Bytes, GIB, MIB};

#[allow(dead_code)]
enum Status {
    Ok,
    Warn,
    Fail,
}

fn line(s: Status, title: &str, detail: String) {
    let mark = match s {
        Status::Ok => ok_mark(),
        Status::Warn => warn_mark(),
        Status::Fail => bad_mark(),
    };
    println!("  {mark} {:<20} {}", bold(title), detail);
}

pub fn run() -> Result<()> {
    println!();
    println!(
        "{}",
        bold(format!("Tendril doctor · v{}", env!("CARGO_PKG_VERSION")))
    );
    println!();
    let n = detect_local(true);

    // Platform & accelerator.
    match n.backend {
        Backend::Metal => line(Status::Ok, "Accelerator", format!("{} via Metal{}", n.chip, n.gpu_cores.map(|c| format!(" ({c} GPU cores)")).unwrap_or_default())),
        Backend::Cuda => line(Status::Ok, "Accelerator", format!("{} via CUDA ({} VRAM)", n.chip, n.accel_memory.unwrap_or_default())),
        Backend::Cpu => line(
            Status::Warn,
            "Accelerator",
            format!("none found — {} will run models on the CPU ({} cores). Fine for small models or as a helper node.", n.chip, n.cpu_cores),
        ),
    }

    // Memory.
    let free = n.available_memory.unwrap_or(n.total_memory);
    let free_frac = free.0 as f64 / n.total_memory.0.max(1) as f64;
    line(
        if free_frac < 0.35 {
            Status::Warn
        } else {
            Status::Ok
        },
        "Memory",
        format!(
            "{} total · {} free{}",
            n.total_memory,
            free,
            if free_frac < 0.35 {
                yellow(" — other apps are using most of it; close them before serving")
            } else {
                String::new()
            }
        ),
    );
    line(
        Status::Ok,
        "Model budget",
        format!(
            "{} {}",
            n.usable_memory,
            dim(format!("({})", n.usable_reason))
        ),
    );
    if n.backend == Backend::Metal {
        match n.wired_limit {
            Some(w) => line(
                Status::Ok,
                "GPU memory limit",
                format!("raised to {w} (iogpu.wired_limit_mb)"),
            ),
            None => {
                let keep = if n.total_memory.0 <= 32 * GIB {
                    3584 * MIB
                } else {
                    6 * GIB
                };
                let target = n.total_memory.0.saturating_sub(keep) / MIB;
                line(
                    Status::Warn,
                    "GPU memory limit",
                    format!(
                        "macOS default (~{}). To fit bigger models: {}",
                        n.total_memory.scale(if n.total_memory.0 <= 36 * GIB {
                            2.0 / 3.0
                        } else {
                            0.75
                        }),
                        cyan(format!("sudo sysctl iogpu.wired_limit_mb={target}"))
                    ),
                );
            }
        }
    }

    // Power.
    match n.on_battery {
        Some(true) => line(
            Status::Warn,
            "Power",
            "on battery — laptops throttle; plug in for serving".into(),
        ),
        Some(false) => line(Status::Ok, "Power", "plugged in".into()),
        None => {}
    }

    // Disk.
    let cache = dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("tendril");
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let disk = disks
        .list()
        .iter()
        .filter(|d| cache.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len());
    if let Some(d) = disk {
        let avail = Bytes(d.available_space());
        line(
            if avail.0 < 20 * GIB {
                Status::Warn
            } else {
                Status::Ok
            },
            "Disk",
            format!("{} free for model files at {}", avail, dim(cache.display())),
        );
    }

    // HuggingFace.
    let endpoint = hf_endpoint();
    let token = hf_token();
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(6))
        .build()?;
    match client.head(format!("{endpoint}/api/models")).send() {
        Ok(r) if r.status().is_success() || r.status().as_u16() == 405 => line(
            Status::Ok,
            "HuggingFace",
            format!(
                "reachable{}",
                if token.is_some() { " · token found (gated models OK)".to_string() } else { dim(" · no HF_TOKEN (needed only for gated models like Llama/Gemma)") }
            ),
        ),
        Ok(r) => line(Status::Warn, "HuggingFace", format!("{endpoint} answered {}", r.status())),
        Err(_) => line(
            Status::Warn,
            "HuggingFace",
            format!("{endpoint} unreachable — local models and the built-in catalog still work (--offline)"),
        ),
    }
    println!();
    println!(
        "{}",
        dim("Next: tendril plan <model>   ·   tendril fit <model>   ·   tendril node --probe")
    );
    Ok(())
}
