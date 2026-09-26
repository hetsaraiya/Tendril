//! Multi-row matmul: Tendril band kernel vs widened-GEMM, for batch sizes.
use std::time::Instant;
use tendril_engine::cpu_kernels::{matmul, CpuWeights};
fn main() {
    let (n, k) = (8192usize, 2048usize);
    let w = CpuWeights::Bf16(
        (0..n * k)
            .map(|i| half::bf16::from_f32(((i % 97) as f32 - 48.0) / 500.0).to_bits())
            .collect(),
    );
    for m in [16usize, 32, 64, 128, 256] {
        let x = vec![0.1f32; m * k];
        let mut out = vec![0f32; m * n];
        matmul(&x, m, k, &w, n, &mut out);
        let t = Instant::now();
        for _ in 0..10 {
            matmul(&x, m, k, &w, n, &mut out);
        }
        let a = t.elapsed().as_secs_f64() * 100.0;
        // widened GEMM via candle
        let dev = candle_core::Device::Cpu;
        let xt = candle_core::Tensor::from_slice(&x, (m, k), &dev).unwrap();
        let t = Instant::now();
        for _ in 0..10 {
            let wf = tendril_engine::cpu_kernels::to_f32(&w, n, k);
            let wt = candle_core::Tensor::from_vec(wf, (n, k), &dev).unwrap();
            let _ = xt.matmul(&wt.t().unwrap()).unwrap();
        }
        let b = t.elapsed().as_secs_f64() * 100.0;
        println!("m={m:2}: band {a:6.2} ms   widen+gemm {b:6.2} ms");
    }
}
