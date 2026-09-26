//! Catalog of known hardware so plans can be made for machines you do not
//! have in front of you ("what if I bought a second Mac mini?").
//!
//! Bandwidth figures are vendor peak numbers; the cost model applies an
//! efficiency factor. Measured profiles (`tendril node --probe`) always win.

use crate::hardware::{Backend, NodeProfile, ProfileSource};
use crate::units::Bytes;

#[derive(Clone, Debug)]
pub struct ChipPreset {
    /// Canonical key, e.g. "m4-pro".
    pub key: &'static str,
    pub display: &'static str,
    pub backend: Backend,
    /// Peak memory bandwidth, GB/s (decimal, as vendors quote it).
    pub bandwidth_gbs: f64,
    /// Approximate dense fp16 throughput usable for prefill, TFLOP/s.
    pub tflops: f64,
    /// Default memory in GiB when the user does not specify it.
    pub default_mem_gib: f64,
    /// Discrete accelerator memory is fixed; unified/CPU memory is configurable.
    pub fixed_mem: bool,
    pub gpu_cores: Option<u32>,
    pub cpu_cores: u32,
}

macro_rules! chip {
    ($key:expr, $disp:expr, $be:expr, $bw:expr, $tf:expr, $mem:expr, $fixed:expr, $gpu:expr, $cpu:expr) => {
        ChipPreset {
            key: $key,
            display: $disp,
            backend: $be,
            bandwidth_gbs: $bw,
            tflops: $tf,
            default_mem_gib: $mem,
            fixed_mem: $fixed,
            gpu_cores: $gpu,
            cpu_cores: $cpu,
        }
    };
}

use Backend::*;

pub static CHIPS: &[ChipPreset] = &[
    // Apple Silicon (unified memory; GPU fp16 TFLOP/s approximations)
    chip!("m1", "Apple M1", Metal, 68.0, 2.6, 16.0, false, Some(8), 8),
    chip!(
        "m1-pro",
        "Apple M1 Pro",
        Metal,
        200.0,
        5.2,
        16.0,
        false,
        Some(16),
        10
    ),
    chip!(
        "m1-max",
        "Apple M1 Max",
        Metal,
        400.0,
        10.4,
        32.0,
        false,
        Some(32),
        10
    ),
    chip!(
        "m1-ultra",
        "Apple M1 Ultra",
        Metal,
        800.0,
        21.0,
        64.0,
        false,
        Some(64),
        20
    ),
    chip!(
        "m2",
        "Apple M2",
        Metal,
        100.0,
        3.6,
        16.0,
        false,
        Some(10),
        8
    ),
    chip!(
        "m2-pro",
        "Apple M2 Pro",
        Metal,
        200.0,
        6.8,
        16.0,
        false,
        Some(19),
        12
    ),
    chip!(
        "m2-max",
        "Apple M2 Max",
        Metal,
        400.0,
        13.6,
        32.0,
        false,
        Some(38),
        12
    ),
    chip!(
        "m2-ultra",
        "Apple M2 Ultra",
        Metal,
        800.0,
        27.2,
        64.0,
        false,
        Some(76),
        24
    ),
    chip!(
        "m3",
        "Apple M3",
        Metal,
        100.0,
        4.1,
        16.0,
        false,
        Some(10),
        8
    ),
    chip!(
        "m3-pro",
        "Apple M3 Pro",
        Metal,
        150.0,
        7.4,
        18.0,
        false,
        Some(18),
        12
    ),
    chip!(
        "m3-max",
        "Apple M3 Max",
        Metal,
        400.0,
        16.4,
        36.0,
        false,
        Some(40),
        16
    ),
    chip!(
        "m3-ultra",
        "Apple M3 Ultra",
        Metal,
        819.0,
        28.0,
        96.0,
        false,
        Some(60),
        28
    ),
    chip!(
        "m4",
        "Apple M4",
        Metal,
        120.0,
        4.3,
        16.0,
        false,
        Some(10),
        10
    ),
    chip!(
        "m4-pro",
        "Apple M4 Pro",
        Metal,
        273.0,
        9.2,
        24.0,
        false,
        Some(20),
        14
    ),
    chip!(
        "m4-max",
        "Apple M4 Max",
        Metal,
        546.0,
        18.4,
        36.0,
        false,
        Some(40),
        16
    ),
    chip!(
        "m5",
        "Apple M5",
        Metal,
        153.0,
        5.7,
        16.0,
        false,
        Some(10),
        10
    ),
    chip!(
        "m5-pro",
        "Apple M5 Pro",
        Metal,
        307.0,
        11.0,
        24.0,
        false,
        Some(20),
        15
    ),
    chip!(
        "m5-max",
        "Apple M5 Max",
        Metal,
        614.0,
        22.0,
        36.0,
        false,
        Some(40),
        18
    ),
    // NVIDIA (VRAM is fixed)
    chip!(
        "rtx3060",
        "NVIDIA RTX 3060 12GB",
        Cuda,
        360.0,
        25.0,
        12.0,
        true,
        None,
        8
    ),
    chip!(
        "rtx3080",
        "NVIDIA RTX 3080 10GB",
        Cuda,
        760.0,
        59.0,
        10.0,
        true,
        None,
        8
    ),
    chip!(
        "rtx3090",
        "NVIDIA RTX 3090",
        Cuda,
        936.0,
        71.0,
        24.0,
        true,
        None,
        8
    ),
    chip!(
        "rtx4060ti",
        "NVIDIA RTX 4060 Ti 16GB",
        Cuda,
        288.0,
        44.0,
        16.0,
        true,
        None,
        8
    ),
    chip!(
        "rtx4070",
        "NVIDIA RTX 4070",
        Cuda,
        504.0,
        58.0,
        12.0,
        true,
        None,
        8
    ),
    chip!(
        "rtx4080",
        "NVIDIA RTX 4080",
        Cuda,
        717.0,
        97.0,
        16.0,
        true,
        None,
        8
    ),
    chip!(
        "rtx4090",
        "NVIDIA RTX 4090",
        Cuda,
        1008.0,
        165.0,
        24.0,
        true,
        None,
        16
    ),
    chip!(
        "rtx5090",
        "NVIDIA RTX 5090",
        Cuda,
        1792.0,
        209.0,
        32.0,
        true,
        None,
        16
    ),
    chip!("t4", "NVIDIA T4", Cuda, 320.0, 65.0, 16.0, true, None, 8),
    chip!("l4", "NVIDIA L4", Cuda, 300.0, 121.0, 24.0, true, None, 8),
    chip!(
        "a10",
        "NVIDIA A10",
        Cuda,
        600.0,
        125.0,
        24.0,
        true,
        None,
        16
    ),
    chip!(
        "a100-40",
        "NVIDIA A100 40GB",
        Cuda,
        1555.0,
        312.0,
        40.0,
        true,
        None,
        32
    ),
    chip!(
        "a100",
        "NVIDIA A100 80GB",
        Cuda,
        2039.0,
        312.0,
        80.0,
        true,
        None,
        32
    ),
    chip!(
        "h100",
        "NVIDIA H100 80GB",
        Cuda,
        3350.0,
        990.0,
        80.0,
        true,
        None,
        32
    ),
    // CPU-only machines
    chip!(
        "cpu",
        "Generic x86 CPU (DDR4)",
        Cpu,
        40.0,
        0.8,
        16.0,
        false,
        None,
        8
    ),
    chip!(
        "cpu-ddr5",
        "Generic x86 CPU (DDR5)",
        Cpu,
        70.0,
        1.5,
        32.0,
        false,
        None,
        12
    ),
    chip!(
        "server-cpu",
        "Server CPU (8-ch DDR5)",
        Cpu,
        300.0,
        4.0,
        256.0,
        false,
        None,
        64
    ),
];

pub fn find_chip(key: &str) -> Option<&'static ChipPreset> {
    let k = normalize(key);
    CHIPS.iter().find(|c| normalize(c.key) == k).or_else(|| {
        // Also accept display names: "Apple M4 Pro", "RTX 4090", "4090".
        CHIPS.iter().find(|c| {
            let d = normalize(c.display);
            d == k || d.ends_with(&k) && k.len() >= 4
        })
    })
}

/// Match a detected marketing name ("Apple M4 Pro", "NVIDIA GeForce RTX 4090").
pub fn match_detected(name: &str) -> Option<&'static ChipPreset> {
    let n = normalize(name).replace("geforce", "");
    // Longest key first so "m4pro" wins over "m4".
    let mut best: Option<&'static ChipPreset> = None;
    for c in CHIPS.iter() {
        let key = normalize(c.key);
        if key.starts_with("cpu") || key == "servercpu" {
            continue;
        }
        let hit = n.ends_with(&key) || n.contains(&key.to_string()) && is_boundary(&n, &key);
        if hit && best.is_none_or(|b| normalize(b.key).len() < key.len()) {
            best = Some(c);
        }
    }
    best
}

fn is_boundary(hay: &str, needle: &str) -> bool {
    // "applem4pro" contains "m4" but also "m4pro"; prefer the longest match,
    // which the caller does. Here we only reject matches followed by a digit
    // ("m1" inside "m10").
    if let Some(pos) = hay.find(needle) {
        let after = hay[pos + needle.len()..].chars().next();
        !matches!(after, Some(c) if c.is_ascii_digit())
    } else {
        false
    }
}

fn normalize(s: &str) -> String {
    s.to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

/// Build a node from a preset spec like "m4", "m4-pro:48", "rtx4090", "cpu:64gb".
pub fn node_from_spec(name: &str, spec: &str) -> Result<NodeProfile, String> {
    let (chip_s, mem_s) = match spec.split_once(':') {
        Some((a, b)) => (a, Some(b)),
        None => (spec, None),
    };
    let chip = find_chip(chip_s).ok_or_else(|| {
        format!(
            "unknown hardware '{chip_s}'. Try one of: {}",
            CHIPS.iter().map(|c| c.key).collect::<Vec<_>>().join(", ")
        )
    })?;
    let mem = match mem_s {
        Some(m) => {
            Bytes::parse(m).ok_or_else(|| format!("cannot parse memory '{m}' (try 16gb)"))?
        }
        None => Bytes::gib(chip.default_mem_gib),
    };
    if chip.fixed_mem && mem_s.is_some() && (mem.as_gib() - chip.default_mem_gib).abs() > 0.5 {
        // Allow e.g. a100:40 even though a separate preset exists; just note it.
    }
    Ok(NodeProfile::from_preset(name, chip, mem))
}

impl NodeProfile {
    pub fn from_preset(name: &str, chip: &ChipPreset, mem: Bytes) -> NodeProfile {
        let (total, accel) = if chip.fixed_mem {
            // Discrete GPU: assume a host with twice the VRAM of system RAM.
            (mem.times(2).max(Bytes::gib(16.0)), Some(mem))
        } else {
            (mem, None)
        };
        let mut n = NodeProfile {
            name: name.to_string(),
            chip: chip.display.to_string(),
            chip_key: Some(chip.key.to_string()),
            backend: chip.backend,
            os: match chip.backend {
                Metal => "macos".into(),
                _ => "linux".into(),
            },
            arch: match chip.backend {
                Metal => "aarch64".into(),
                _ => "x86_64".into(),
            },
            cpu_cores: chip.cpu_cores,
            gpu_cores: chip.gpu_cores,
            total_memory: total,
            accel_memory: accel,
            available_memory: None,
            usable_memory: Bytes::ZERO,
            usable_reason: String::new(),
            wired_limit: None,
            bandwidth_gbs: chip.bandwidth_gbs,
            tflops: chip.tflops,
            source: ProfileSource::Preset(chip.key.to_string()),
            measured_bandwidth_gbs: None,
            on_battery: None,
            address: None,
        };
        n.recompute_usable();
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_names() {
        assert_eq!(match_detected("Apple M4 Pro").unwrap().key, "m4-pro");
        assert_eq!(match_detected("Apple M4").unwrap().key, "m4");
        assert_eq!(match_detected("Apple M1 Max").unwrap().key, "m1-max");
        assert_eq!(
            match_detected("NVIDIA GeForce RTX 4090").unwrap().key,
            "rtx4090"
        );
        assert!(match_detected("Intel Xeon").is_none());
    }

    #[test]
    fn specs() {
        let n = node_from_spec("a", "m4:16gb").unwrap();
        assert_eq!(n.total_memory, Bytes::gib(16.0));
        assert!(n.usable_memory < n.total_memory);
        let g = node_from_spec("g", "rtx4090").unwrap();
        assert_eq!(g.accel_memory, Some(Bytes::gib(24.0)));
        assert!(node_from_spec("x", "nonsense").is_err());
    }
}
