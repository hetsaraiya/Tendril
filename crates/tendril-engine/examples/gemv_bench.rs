//! Compare Tendril's CPU kernels with candle's generic matmul.
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};
use std::time::Instant;
use tendril_engine::cpu_kernels::{matmul, quantize_q8, CpuWeights};

fn time(f: &mut dyn FnMut()) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..20 {
        f();
    }
    t.elapsed().as_secs_f64() * 1000.0 / 20.0
}

fn main() -> anyhow::Result<()> {
    let (k, n) = (4096usize, 14336usize);
    let dev = Device::Cpu;
    let w = Tensor::randn(0f32, 0.02, (n, k), &dev)?;
    let wv: Vec<f32> = w.flatten_all()?.to_vec1()?;
    let x = Tensor::randn(0f32, 1.0, (1, k), &dev)?;
    let xv: Vec<f32> = x.flatten_all()?.to_vec1()?;
    let mut out = vec![0f32; n];
    let lin = candle_nn::Linear::new(w.clone(), None);
    let ms = time(&mut || {
        lin.forward(&x).unwrap();
    });
    println!(
        "candle f32        {ms:7.2} ms  {:6.1} GB/s",
        (n * k * 4) as f64 / ms / 1e6
    );
    let qm = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0)?)?;
    let ms = time(&mut || {
        qm.forward(&x).unwrap();
    });
    println!(
        "candle q8_0       {ms:7.2} ms  {:6.1} GB/s",
        (n * k) as f64 * 1.0625 / ms / 1e6
    );
    for cw in [
        CpuWeights::F32(wv.clone()),
        CpuWeights::Bf16(
            wv.iter()
                .map(|v| half::bf16::from_f32(*v).to_bits())
                .collect(),
        ),
        CpuWeights::F16(
            wv.iter()
                .map(|v| half::f16::from_f32(*v).to_bits())
                .collect(),
        ),
        CpuWeights::Q8(quantize_q8(&wv)),
    ] {
        let ms = time(&mut || matmul(&xv, 1, k, &cw, n, &mut out));
        println!(
            "tendril {:<9} {ms:7.2} ms  {:6.1} GB/s",
            cw.label(),
            cw.bytes() as f64 / ms / 1e6
        );
    }
    Ok(())
}
