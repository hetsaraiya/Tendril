//! Measure how fast this machine runs Tendril's own kernels.
//!
//! Two numbers drive the planner: how many bytes of weights per second the
//! decode kernel streams (a matrix-vector product over a matrix far larger
//! than any cache), and how many FLOP/s a prefill-sized matrix multiply
//! reaches. Both run on the device the machine will actually use.

use crate::linear::Linear;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct DeviceBench {
    /// Weight bytes streamed per second by the decode kernel, GB/s.
    pub weight_gbs: f64,
    /// Matrix-multiply throughput, TFLOP/s.
    pub tflops: f64,
    pub seconds: f64,
}

/// Run `f` until `budget` has passed (at least `min` times) and return the
/// fastest run. The fastest is the machine's real capability; slower runs
/// are noise from other programs.
fn fastest(min: usize, budget: Duration, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?; // warm up: page in, compile kernels, spin up threads
    let start = Instant::now();
    let mut best = f64::INFINITY;
    let mut runs = 0;
    while runs < min || start.elapsed() < budget {
        let t = Instant::now();
        f()?;
        best = best.min(t.elapsed().as_secs_f64());
        runs += 1;
        if runs >= 200 {
            break;
        }
    }
    Ok(best)
}

/// Wait for the device: read the (small) result back.
fn sync(y: &Tensor) -> Result<()> {
    let _ = y
        .flatten_all()?
        .to_dtype(DType::F32)?
        .max(0)?
        .to_scalar::<f32>()?;
    Ok(())
}

/// Benchmark `device`. Takes a few seconds; `step` reports progress.
pub fn run(device: &Device, step: &mut dyn FnMut(&str)) -> Result<DeviceBench> {
    let t0 = Instant::now();
    let act = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    // Decode: one token through a bf16 weight much larger than any cache
    // (256 MiB on CPU, 1 GiB on a GPU).
    step("decode speed (streaming weights)");
    let (n, k) = if device.is_cpu() {
        (16384, 8192)
    } else {
        (32768, 16384)
    };
    let w = Linear::random_bf16(n, k, device)?;
    let x = Tensor::ones((1, k), act, device)?;
    let secs = fastest(5, Duration::from_millis(1500), || sync(&w.forward(&x)?))?;
    let weight_gbs = (n * k * 2) as f64 / secs / 1e9;
    drop(w);

    // Prefill: a batch of prompt tokens through a square weight.
    step("prefill speed (matrix multiply)");
    let (m, d) = if device.is_cpu() {
        (512, 4096)
    } else {
        (2048, 8192)
    };
    let w = Linear::random_bf16(d, d, device)?;
    let x = Tensor::ones((m, d), act, device)?;
    let secs = fastest(3, Duration::from_millis(1500), || sync(&w.forward(&x)?))?;
    let tflops = (2 * m * d * d) as f64 / secs / 1e12;
    Ok(DeviceBench {
        weight_gbs,
        tflops,
        seconds: t0.elapsed().as_secs_f64(),
    })
}
