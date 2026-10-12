//! The startup benchmark: measure this machine once per hardware setup.

use crate::ui::*;
use anyhow::Result;
use std::io::Write;
use tendril_core::hardware::{fingerprint, probe_memory_bandwidth, save_benchmark, Benchmark};

/// Make sure this machine has a benchmark for `device`, measuring it if
/// the hardware is new (or `force`). Prints one line either way.
pub fn ensure(device: &str, force: bool) -> Result<Benchmark> {
    let p = tendril_cluster::coordinator::this_machine(None, device, None)?;
    if !force {
        if let Some(b) = tendril_core::hardware::saved_benchmark(&p) {
            println!(
                "{} {} {}",
                dim("·"),
                summary(&b),
                dim(format!(
                    "(benchmarked {}; `tendril node --probe` measures again)",
                    b.measured_at
                ))
            );
            return Ok(b);
        }
    }
    print!(
        "{} Benchmarking this machine{} — ",
        dim("·"),
        if force {
            ""
        } else {
            " (first run on this hardware, about 10 s)"
        }
    );
    std::io::stdout().flush().ok();
    let dev = tendril_engine::device::device_from_name(device)?;
    let r = tendril_engine::hwbench::run(&dev, &mut |step| {
        print!("{} ", dim(step));
        std::io::stdout().flush().ok();
    })?;
    let b = Benchmark {
        fingerprint: fingerprint(&p),
        weight_gbs: r.weight_gbs,
        tflops: r.tflops,
        copy_gbs: probe_memory_bandwidth(),
        device: tendril_engine::device::device_label(&dev).into(),
        measured_at: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
        seconds: r.seconds,
    };
    save_benchmark(&b);
    println!("{}", ok_mark());
    println!("{} {}", ok_mark(), summary(&b));
    Ok(b)
}

fn summary(b: &Benchmark) -> String {
    format!(
        "{}: decode streams {} of weights/s · prefill {} TFLOP/s",
        b.device,
        bold(format!("{:.1} GB", b.weight_gbs)),
        bold(format!("{:.2}", b.tflops)),
    )
}
