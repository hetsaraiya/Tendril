//! Decode throughput on a random ~0.5B-parameter Llama-shaped model.
use candle_core::{DType, Device};
use std::time::Instant;
use tendril_engine::generate::{GenerateRequest, LocalModel, ModelFiles};
use tendril_engine::linear::WeightFormat;
use tendril_engine::sampler::SamplingParams;

fn main() -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join("tendril-decode-bench");
    if !dir.join("model.safetensors").exists() {
        tendril_engine::testing::write_tiny_llama(&dir, 16, 2048, 7, DType::BF16)?;
    }
    for fmt in [WeightFormat::Native, WeightFormat::Q8_0] {
        let t = Instant::now();
        let mut m = LocalModel::load(ModelFiles::new(&dir)?, Device::Cpu, fmt, None)?;
        let load = t.elapsed().as_secs_f64();
        let prompt: Vec<u32> = (0..200).map(|i| 5 + (i * 7 % 250) as u32).collect();
        let timing = m.generate(
            GenerateRequest { prompt, params: SamplingParams::greedy(), max_tokens: 64, stop: vec![] },
            |_| true,
        )?;
        let gb = m.weight_bytes() as f64 / 1e9;
        println!(
            "{:7} weights {:.2} GB  load {:.1}s  prefill {:.0} tok/s  decode {:.1} tok/s  ({:.1} GB/s effective)",
            fmt.label(),
            gb,
            load,
            timing.prefill_tps(),
            timing.decode_tps(),
            timing.decode_tps() * gb
        );
    }
    Ok(())
}
