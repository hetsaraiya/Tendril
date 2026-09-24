use std::time::Instant;
use tendril_engine::cpu_kernels::{matmul, CpuWeights};
fn main() {
    for (n, k, count) in [(2048usize, 2048usize, 64usize), (4096, 2048, 32), (8192, 2048, 16), (14336, 4096, 4)] {
        let ws: Vec<CpuWeights> = (0..count).map(|i| CpuWeights::Bf16(vec![0x3f80u16 + i as u16; n * k])).collect();
        let x = vec![0.5f32; k];
        let mut out = vec![0f32; n];
        for w in &ws { matmul(&x, 1, k, w, n, &mut out); }
        let rounds = 5;
        let t = Instant::now();
        for _ in 0..rounds { for w in &ws { matmul(&x, 1, k, w, n, &mut out); } }
        let us = t.elapsed().as_secs_f64() * 1e6 / (rounds * count) as f64;
        println!("cold {n}x{k}: {us:.0} us  {:.1} GB/s", (n * k * 2) as f64 / us / 1e3);
    }
}
