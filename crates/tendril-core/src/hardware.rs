//! Node hardware profiles and local detection.

use crate::presets;
use crate::units::{Bytes, GIB, MIB};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Apple Silicon GPU through Metal.
    Metal,
    /// NVIDIA GPU through CUDA.
    Cuda,
    /// Plain CPU execution.
    Cpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Metal => "Metal",
            Backend::Cuda => "CUDA",
            Backend::Cpu => "CPU",
        }
    }
    /// Fixed runtime overhead (context, command buffers, allocator slack).
    pub fn runtime_overhead(self) -> Bytes {
        match self {
            Backend::Metal => Bytes(384 * MIB),
            Backend::Cuda => Bytes(640 * MIB),
            Backend::Cpu => Bytes(192 * MIB),
        }
    }
    /// Fraction of peak memory bandwidth realistically achieved while
    /// streaming weights during decode.
    pub fn bandwidth_efficiency(self) -> f64 {
        match self {
            Backend::Metal => 0.78,
            Backend::Cuda => 0.82,
            Backend::Cpu => 0.65,
        }
    }
    pub fn compute_efficiency(self) -> f64 {
        match self {
            Backend::Metal => 0.55,
            Backend::Cuda => 0.5,
            Backend::Cpu => 0.45,
        }
    }
    /// Fixed per-layer dispatch cost during decode, milliseconds.
    pub fn per_layer_overhead_ms(self) -> f64 {
        match self {
            Backend::Metal => 0.045,
            Backend::Cuda => 0.03,
            Backend::Cpu => 0.02,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProfileSource {
    Detected,
    Preset(String),
    Manual,
}

/// Everything the planner needs to know about one machine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeProfile {
    pub name: String,
    pub chip: String,
    #[serde(default)]
    pub chip_key: Option<String>,
    pub backend: Backend,
    pub os: String,
    pub arch: String,
    pub cpu_cores: u32,
    #[serde(default)]
    pub gpu_cores: Option<u32>,
    /// System RAM (unified memory on Apple Silicon).
    pub total_memory: Bytes,
    /// Discrete accelerator memory (VRAM), if any.
    #[serde(default)]
    pub accel_memory: Option<Bytes>,
    /// Memory currently free, when detected live.
    #[serde(default)]
    pub available_memory: Option<Bytes>,
    /// Memory the planner may use for weights + KV + buffers.
    pub usable_memory: Bytes,
    #[serde(default)]
    pub usable_reason: String,
    /// macOS `iogpu.wired_limit_mb` if set by the user.
    #[serde(default)]
    pub wired_limit: Option<Bytes>,
    /// Peak memory bandwidth, GB/s.
    pub bandwidth_gbs: f64,
    /// Approximate fp16 TFLOP/s.
    pub tflops: f64,
    pub source: ProfileSource,
    #[serde(default)]
    pub measured_bandwidth_gbs: Option<f64>,
    #[serde(default)]
    pub on_battery: Option<bool>,
    /// Network address when this node belongs to a live cluster.
    #[serde(default)]
    pub address: Option<String>,
}

impl NodeProfile {
    /// Effective bandwidth used by the cost model (GB/s).
    pub fn effective_bandwidth_gbs(&self) -> f64 {
        match self.measured_bandwidth_gbs {
            // A measured CPU memcpy already reflects achievable bandwidth.
            Some(m) if self.backend == Backend::Cpu => m * 0.9,
            _ => self.bandwidth_gbs * self.backend.bandwidth_efficiency(),
        }
    }

    pub fn effective_tflops(&self) -> f64 {
        self.tflops * self.backend.compute_efficiency()
    }

    /// Recompute the memory budget from the hardware facts.
    pub fn recompute_usable(&mut self) {
        let (usable, reason) = match self.backend {
            Backend::Metal => {
                let total = self.total_memory;
                if let Some(w) = self.wired_limit {
                    (
                        w,
                        format!("macOS GPU wired limit set to {w} (iogpu.wired_limit_mb)"),
                    )
                } else {
                    let frac = if total.0 <= 36 * GIB { 2.0 / 3.0 } else { 0.75 };
                    (
                        total.scale(frac),
                        format!(
                            "macOS lets the GPU wire ~{:.0}% of {total} by default",
                            frac * 100.0
                        ),
                    )
                }
            }
            Backend::Cuda => {
                let vram = self.accel_memory.unwrap_or(self.total_memory);
                (
                    vram.saturating_sub(Bytes(600 * MIB)),
                    format!("{vram} VRAM minus CUDA context"),
                )
            }
            Backend::Cpu => {
                let total = self.total_memory;
                let reserve = Bytes((total.0 / 8).max(2 * GIB));
                (
                    total.saturating_sub(reserve),
                    format!("{total} RAM minus {reserve} for the OS"),
                )
            }
        };
        self.usable_memory = usable;
        self.usable_reason = reason;
        if let Some(avail) = self.available_memory {
            // For unified/CPU memory, other apps compete for the same pool.
            if self.backend != Backend::Cuda && avail < self.usable_memory {
                self.usable_memory = avail.saturating_sub(Bytes(512 * MIB));
                self.usable_reason = format!(
                    "only {avail} free right now (other apps are using memory; close them to free more)"
                );
            }
        }
    }

    pub fn is_laptop_class(&self) -> bool {
        matches!(self.backend, Backend::Metal) && !self.chip.contains("Ultra")
    }
}

/// Detect the local machine.
pub fn detect_local(consider_free_memory: bool) -> NodeProfile {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
    let total = Bytes(sys.total_memory());
    let avail = Bytes(sys.available_memory());
    let cpu_cores = sys.cpus().len().max(1) as u32;
    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .unwrap_or_default();
    let hostname = System::host_name().unwrap_or_else(|| "this-machine".into());
    let short_host = hostname.split('.').next().unwrap_or(&hostname).to_string();
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    let mut node = NodeProfile {
        name: short_host,
        chip: if cpu_brand.is_empty() {
            "Unknown CPU".into()
        } else {
            cpu_brand.clone()
        },
        chip_key: None,
        backend: Backend::Cpu,
        os: os.clone(),
        arch: arch.clone(),
        cpu_cores,
        gpu_cores: None,
        total_memory: total,
        accel_memory: None,
        available_memory: if consider_free_memory {
            Some(avail)
        } else {
            None
        },
        usable_memory: Bytes::ZERO,
        usable_reason: String::new(),
        wired_limit: None,
        bandwidth_gbs: 40.0,
        tflops: 0.8,
        source: ProfileSource::Detected,
        measured_bandwidth_gbs: None,
        on_battery: detect_battery(),
        address: None,
    };

    if os == "macos" && arch == "aarch64" {
        let brand = sysctl("machdep.cpu.brand_string").unwrap_or(cpu_brand);
        node.chip = brand.clone();
        node.backend = Backend::Metal;
        if let Some(p) = presets::match_detected(&brand) {
            node.chip_key = Some(p.key.to_string());
            node.bandwidth_gbs = p.bandwidth_gbs;
            node.tflops = p.tflops;
            node.gpu_cores = p.gpu_cores;
        } else {
            node.bandwidth_gbs = 100.0;
            node.tflops = 4.0;
        }
        if let Some(cores) = detect_apple_gpu_cores() {
            node.gpu_cores = Some(cores);
        }
        if let Some(limit) = sysctl("iogpu.wired_limit_mb").and_then(|s| s.parse::<u64>().ok()) {
            if limit > 0 {
                node.wired_limit = Some(Bytes(limit * MIB));
            }
        }
    } else if let Some((name, vram)) = detect_nvidia() {
        node.chip = name.clone();
        node.backend = Backend::Cuda;
        node.accel_memory = Some(vram);
        node.available_memory = None;
        if let Some(p) = presets::match_detected(&name) {
            node.chip_key = Some(p.key.to_string());
            node.bandwidth_gbs = p.bandwidth_gbs;
            node.tflops = p.tflops;
        } else {
            node.bandwidth_gbs = 500.0;
            node.tflops = 40.0;
        }
    } else {
        // Rough CPU compute estimate: cores * ~40 GFLOP/s (AVX2/NEON fp32).
        node.tflops = (cpu_cores as f64 * 0.04).max(0.2);
        node.bandwidth_gbs = if total.0 >= 128 * GIB { 120.0 } else { 40.0 };
    }
    node.recompute_usable();
    node
}

fn sysctl(key: &str) -> Option<String> {
    let out = run_with_timeout("sysctl", &["-n", key], Duration::from_secs(2))?;
    let s = out.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn detect_apple_gpu_cores() -> Option<u32> {
    let out = run_with_timeout(
        "system_profiler",
        &["SPDisplaysDataType"],
        Duration::from_secs(5),
    )?;
    out.lines()
        .find_map(|l| l.trim().strip_prefix("Total Number of Cores:"))
        .and_then(|v| v.trim().parse().ok())
}

fn detect_nvidia() -> Option<(String, Bytes)> {
    let out = run_with_timeout(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        Duration::from_secs(5),
    )?;
    let line = out.lines().next()?;
    let (name, mem) = line.rsplit_once(',')?;
    let mib: u64 = mem.trim().parse().ok()?;
    Some((name.trim().to_string(), Bytes(mib * MIB)))
}

fn detect_battery() -> Option<bool> {
    if cfg!(target_os = "macos") {
        let out = run_with_timeout("pmset", &["-g", "batt"], Duration::from_secs(2))?;
        Some(out.contains("Battery Power"))
    } else if cfg!(target_os = "linux") {
        let dir = std::fs::read_dir("/sys/class/power_supply").ok()?;
        let mut saw_battery = false;
        for e in dir.flatten() {
            let p = e.path();
            let kind = std::fs::read_to_string(p.join("type")).unwrap_or_default();
            if kind.trim() == "Battery" {
                saw_battery = true;
                let st = std::fs::read_to_string(p.join("status")).unwrap_or_default();
                if st.trim() == "Discharging" {
                    return Some(true);
                }
            }
        }
        saw_battery.then_some(false)
    } else {
        None
    }
}

fn run_with_timeout(cmd: &str, args: &[&str], timeout: Duration) -> Option<String> {
    use std::process::Stdio;
    let mut child = Command::new(cmd)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let mut s = String::new();
                use std::io::Read;
                child.stdout.take()?.read_to_string(&mut s).ok()?;
                return Some(s);
            }
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(_) => return None,
        }
    }
}

/// Measure achievable memory bandwidth with a multi-threaded copy.
/// Returns GB/s (decimal). Takes well under a second.
pub fn probe_memory_bandwidth() -> f64 {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);
    let per_thread = (256 * MIB as usize) / threads;
    let mut srcs: Vec<Vec<u8>> = (0..threads).map(|i| vec![i as u8; per_thread]).collect();
    let mut dsts: Vec<Vec<u8>> = (0..threads).map(|_| vec![0u8; per_thread]).collect();
    let mut best = 0.0f64;
    for _ in 0..4 {
        let start = Instant::now();
        std::thread::scope(|s| {
            for (src, dst) in srcs.iter_mut().zip(dsts.iter_mut()) {
                s.spawn(move || {
                    dst.copy_from_slice(src);
                    std::hint::black_box(&dst[0]);
                });
            }
        });
        let secs = start.elapsed().as_secs_f64();
        // A copy reads and writes every byte.
        let gbs = (2 * per_thread * threads) as f64 / secs / 1e9;
        best = best.max(gbs);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_runs() {
        let n = detect_local(true);
        assert!(n.total_memory.0 > 0);
        assert!(n.usable_memory <= n.total_memory);
    }

    #[test]
    fn mac_usable() {
        let n = presets::node_from_spec("a", "m4:16").unwrap();
        // 2/3 of 16 GiB
        assert!((n.usable_memory.as_gib() - 10.67).abs() < 0.05);
        let big = presets::node_from_spec("b", "m4-max:64").unwrap();
        assert!((big.usable_memory.as_gib() - 48.0).abs() < 0.05);
    }
}
