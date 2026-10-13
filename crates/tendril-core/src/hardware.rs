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
    /// Weight-streaming bandwidth measured through Tendril's own decode
    /// kernels on this machine (GB/s); replaces the spec-sheet estimate.
    #[serde(default)]
    pub measured_bandwidth_gbs: Option<f64>,
    /// Matrix-multiply throughput measured the same way (TFLOP/s).
    #[serde(default)]
    pub measured_tflops: Option<f64>,
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
            // Measured through the real kernels: already what decode achieves.
            Some(m) if m > 0.0 => m,
            _ => self.bandwidth_gbs * self.backend.bandwidth_efficiency(),
        }
    }

    pub fn effective_tflops(&self) -> f64 {
        match self.measured_tflops {
            Some(t) if t > 0.0 => t,
            _ => self.tflops * self.backend.compute_efficiency(),
        }
    }

    /// Run on `backend` (e.g. a Mac forced onto its CPU): redo the memory
    /// budget and pick up that backend's benchmark.
    pub fn use_backend(&mut self, backend: Backend) {
        if self.backend != backend {
            self.backend = backend;
            self.recompute_usable();
            apply_benchmark(self);
        }
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
}

/// Detect the local machine.
pub fn detect_local(consider_free_memory: bool) -> NodeProfile {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
    let total = Bytes(sys.total_memory());
    // sysinfo undercounts on recent macOS (inactive pages, which apps get
    // back on demand, are left out), so ask vm_stat there.
    let avail = Bytes(
        run_with_timeout("vm_stat", &[], Duration::from_secs(2))
            .and_then(|t| vm_stat_available(&t))
            .unwrap_or_else(|| sys.available_memory()),
    );
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
        available_memory: if consider_free_memory && avail.0 > 0 {
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
        measured_tflops: None,
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
    apply_benchmark(&mut node);
    node
}

/// Bump when the benchmark changes so every machine measures again.
pub const BENCH_VERSION: u32 = 2;

/// A machine's measured speed, kept until its hardware changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Benchmark {
    /// What was measured (see [`fingerprint`]).
    pub fingerprint: String,
    /// Memory read speed while streaming weights, GB/s.
    pub weight_gbs: f64,
    /// Matrix-multiply throughput (prefill), TFLOP/s, on the CPU.
    pub tflops: f64,
    /// Plain memory copy, GB/s (for reference).
    pub copy_gbs: f64,
    /// "CPU", "Metal" or "CUDA".
    pub device: String,
    /// Local date and time of the measurement.
    pub measured_at: String,
    pub seconds: f64,
}

/// Identifies the hardware a benchmark is valid for: the same chip, cores,
/// memory and backend measure the same. A new GPU, more RAM or another OS
/// gives a new fingerprint and a fresh benchmark.
pub fn fingerprint(p: &NodeProfile) -> String {
    format!(
        "v{BENCH_VERSION}|{}|{}|{}|{}|{}c|{}g|{}GiB|{}",
        p.chip,
        p.backend.label(),
        p.os,
        p.arch,
        p.cpu_cores,
        p.gpu_cores.unwrap_or(0),
        (p.total_memory.0 as f64 / GIB as f64).round() as u64,
        p.accel_memory
            .map(|b| format!("{}GiB", (b.0 as f64 / GIB as f64).round() as u64))
            .unwrap_or_default(),
    )
}

fn bench_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("TENDRIL_BENCH_FILE") {
        return Some(p.into());
    }
    Some(dirs::config_dir()?.join("tendril").join("benchmarks.json"))
}

fn load_benchmarks() -> Vec<Benchmark> {
    bench_path()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// This hardware's saved benchmark, if it was measured before.
pub fn saved_benchmark(p: &NodeProfile) -> Option<Benchmark> {
    let fp = fingerprint(p);
    load_benchmarks().into_iter().find(|b| b.fingerprint == fp)
}

/// Remember a benchmark (replacing any earlier one for the same hardware).
pub fn save_benchmark(b: &Benchmark) {
    let Some(path) = bench_path() else { return };
    let mut all = load_benchmarks();
    all.retain(|x| x.fingerprint != b.fingerprint);
    all.push(b.clone());
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    if let Ok(j) = serde_json::to_vec_pretty(&all) {
        let _ = std::fs::write(path, j);
    }
}

/// Use this hardware's measured speed in place of spec-sheet numbers
/// (CPU machines; see [`measure`]).
pub fn apply_benchmark(node: &mut NodeProfile) {
    node.measured_bandwidth_gbs = None;
    node.measured_tflops = None;
    if let Some(b) = saved_benchmark(node) {
        apply_measured_cpu(node, &b);
    }
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

/// Free + reclaimable (inactive, speculative) memory from `vm_stat` output.
fn vm_stat_available(text: &str) -> Option<u64> {
    let page: u64 = text
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let pages = |key: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(key))?;
        line[key.len()..].trim().trim_end_matches('.').parse().ok()
    };
    Some(page * (pages("Pages free:")? + pages("Pages inactive:")? + pages("Pages speculative:")?))
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

/// Measure this machine: memory bandwidth and CPU matrix throughput (a few
/// seconds). GPUs are left to their spec-sheet numbers: each engine has its
/// own kernels, so no single measurement is right for all of them.
pub fn measure(p: &NodeProfile) -> Benchmark {
    let t0 = Instant::now();
    let copy = probe_memory_bandwidth();
    let read = probe_read_bandwidth();
    let tflops = probe_cpu_tflops();
    Benchmark {
        fingerprint: fingerprint(p),
        weight_gbs: read,
        tflops,
        copy_gbs: copy,
        device: "CPU".into(),
        measured_at: String::new(),
        seconds: t0.elapsed().as_secs_f64(),
    }
}

/// Use a CPU benchmark only for machines that run models on the CPU.
pub fn apply_measured_cpu(node: &mut NodeProfile, b: &Benchmark) {
    if node.backend == Backend::Cpu {
        node.measured_bandwidth_gbs = Some(b.weight_gbs);
        node.measured_tflops = Some(b.tflops);
    }
}

/// Multi-threaded sequential reads (what decoding does with weights), GB/s.
pub fn probe_read_bandwidth() -> f64 {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(32);
    let per_thread = (512 * MIB as usize) / threads / 8;
    let data: Vec<Vec<u64>> = (0..threads)
        .map(|i| vec![i as u64 + 1; per_thread])
        .collect();
    let mut best = 0.0f64;
    for _ in 0..4 {
        let start = Instant::now();
        std::thread::scope(|s| {
            for d in &data {
                s.spawn(move || {
                    let mut acc = [0u64; 4];
                    for i in (0..d.len() - d.len() % 4).step_by(4) {
                        acc[0] = acc[0].wrapping_add(d[i]);
                        acc[1] = acc[1].wrapping_add(d[i + 1]);
                        acc[2] = acc[2].wrapping_add(d[i + 2]);
                        acc[3] = acc[3].wrapping_add(d[i + 3]);
                    }
                    std::hint::black_box(acc);
                });
            }
        });
        let gbs = (per_thread * 8 * threads) as f64 / start.elapsed().as_secs_f64() / 1e9;
        best = best.max(gbs);
    }
    best
}

/// fp32 matrix multiply on every core, TFLOP/s (a blocked kernel the
/// compiler vectorizes; engines with tuned kernels do better).
pub fn probe_cpu_tflops() -> f64 {
    const N: usize = 256;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(64);
    let a: Vec<f32> = (0..N * N).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..N * N).map(|i| (i % 5) as f32 * 0.2).collect();
    let mut best = 0.0f64;
    for _ in 0..3 {
        let start = Instant::now();
        let reps = 4;
        std::thread::scope(|s| {
            for _ in 0..threads {
                let (a, b) = (&a, &b);
                s.spawn(move || {
                    let mut c = vec![0f32; N * N];
                    for _ in 0..reps {
                        for i in 0..N {
                            let row = &mut c[i * N..(i + 1) * N];
                            for k in 0..N {
                                let x = a[i * N + k];
                                let bk = &b[k * N..(k + 1) * N];
                                for (r, y) in row.iter_mut().zip(bk) {
                                    *r += x * y;
                                }
                            }
                        }
                    }
                    std::hint::black_box(&c);
                });
            }
        });
        let flops = 2.0 * (N * N * N) as f64 * reps as f64 * threads as f64;
        best = best.max(flops / start.elapsed().as_secs_f64() / 1e12);
    }
    best
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
    fn vm_stat_counts_reclaimable_pages() {
        let t = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:  10.\nPages active: 99.\nPages inactive:  20.\nPages speculative:  2.\n";
        assert_eq!(vm_stat_available(t), Some(16384 * 32));
        assert_eq!(vm_stat_available("garbage"), None);
    }

    #[test]
    fn benchmark_is_reused_until_the_hardware_changes() {
        let dir = std::env::temp_dir().join(format!("tendril-bench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("TENDRIL_BENCH_FILE", dir.join("b.json"));
        let mut p = crate::presets::node_from_spec("box", "m4:16").unwrap();
        p.use_backend(Backend::Cpu);
        let spec_bw = p.effective_bandwidth_gbs();
        save_benchmark(&Benchmark {
            fingerprint: fingerprint(&p),
            weight_gbs: 42.0,
            tflops: 1.5,
            copy_gbs: 60.0,
            device: "CPU".into(),
            measured_at: "now".into(),
            seconds: 1.0,
        });
        apply_benchmark(&mut p);
        assert_eq!(
            p.effective_bandwidth_gbs(),
            42.0,
            "measured replaces the spec sheet"
        );
        assert_eq!(p.effective_tflops(), 1.5);
        // More memory is different hardware: measure again.
        let mut bigger = crate::presets::node_from_spec("box", "m4:32").unwrap();
        bigger.use_backend(Backend::Cpu);
        assert!(bigger.measured_bandwidth_gbs.is_none());
        // GPUs keep their spec-sheet numbers (each engine has its own kernels).
        p.use_backend(Backend::Metal);
        assert!(p.measured_bandwidth_gbs.is_none());
        assert_ne!(p.effective_bandwidth_gbs(), spec_bw);
        std::env::remove_var("TENDRIL_BENCH_FILE");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn cpu_probes_measure_something() {
        assert!(probe_cpu_tflops() > 0.0);
        assert!(probe_read_bandwidth() > 0.0);
    }

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
