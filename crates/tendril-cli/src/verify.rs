//! `tendril verify`: prove that splitting a model changes nothing.

use crate::models::ensure_local;
use crate::ui::*;
use anyhow::{bail, Result};
use clap::Args;
use std::sync::Arc;
use std::time::Instant;
use tendril_engine::config::ModelConfig;
use tendril_engine::device::{activation_dtype, device_from_name};
use tendril_engine::linear::WeightFormat;
use tendril_engine::model::{LoadOptions, Stage, StageInput, StageOutput, StageSpec};
use tendril_engine::weights::WeightStore;

#[derive(Args, Debug)]
pub struct VerifyArgs {
    pub model: String,
    /// Number of pipeline stages to compare against a single stage.
    #[arg(long, default_value_t = 2)]
    pub stages: usize,
    #[arg(long, default_value = "auto")]
    pub device: String,
    #[arg(long, short = 'q', default_value = "native")]
    pub quantize: String,
    #[arg(long)]
    pub offline: bool,
}

fn run_pipeline(stages: &mut [Stage], ids: &[u32]) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::new();
    for (pos, &t) in ids.iter().enumerate() {
        let mut input = StageInput::Tokens(vec![t]);
        for s in stages.iter_mut() {
            match s.forward(1, pos, input, true)? {
                StageOutput::Hidden(h) => input = StageInput::Hidden(h),
                StageOutput::Logits(l) => {
                    out.push(l);
                    break;
                }
                StageOutput::Nothing | StageOutput::AllLogits(_) => bail!("no logits"),
            }
        }
    }
    Ok(out)
}

pub fn run(a: VerifyArgs) -> Result<()> {
    let (dir, name) = ensure_local(&a.model, !a.offline)?;
    let format =
        WeightFormat::parse(&a.quantize).ok_or_else(|| anyhow::anyhow!("unknown --quantize"))?;
    let cfg = Arc::new(ModelConfig::from_file(&dir.join("config.json"))?);
    let tok = tendril_engine::tokenizer::Tok::from_dir(&dir, &cfg.eos_token_ids, cfg.bos_token_id)?;
    let ws = WeightStore::open_dir(&dir)?;
    let device = device_from_name(&a.device)?;
    let dtype = activation_dtype(&device, &cfg.torch_dtype);
    let opts = LoadOptions {
        format,
        device,
        dtype,
    };
    let n = cfg.num_layers;
    let k = a.stages.clamp(2, n);
    println!(
        "{} Verifying {} split into {k} stages against one stage",
        cyan("◌"),
        bold(&name)
    );
    let ids = tok.encode_prompt(
        "The quick brown fox jumps over the lazy dog. Tendril splits models across machines.",
    )?;
    let t0 = Instant::now();
    let mut whole = vec![Stage::load(
        cfg.clone(),
        &ws,
        StageSpec::whole(&cfg),
        &opts,
    )?];
    let a_logits = run_pipeline(&mut whole, &ids)?;
    drop(whole);
    let mut specs = Vec::new();
    for i in 0..k {
        let s = i * n / k;
        let e = (i + 1) * n / k;
        specs.push(StageSpec {
            layer_start: s,
            layer_end: e,
            embed: i == 0,
            head: i == k - 1,
        });
    }
    let mut split: Vec<Stage> = specs
        .iter()
        .map(|s| Stage::load(cfg.clone(), &ws, *s, &opts))
        .collect::<Result<_>>()?;
    let b_logits = run_pipeline(&mut split, &ids)?;
    let mut max_diff = 0f32;
    let mut agree = 0;
    for (x, y) in a_logits.iter().zip(&b_logits) {
        for (p, q) in x.iter().zip(y) {
            max_diff = max_diff.max((p - q).abs());
        }
        let am = |v: &Vec<f32>| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        agree += usize::from(am(x) == am(y));
    }
    println!(
        "  {} positions compared · max |Δlogit| = {max_diff:.2e} · next-token agreement {agree}/{} · {:.1} s",
        ids.len(),
        ids.len(),
        t0.elapsed().as_secs_f64()
    );
    for (i, s) in specs.iter().enumerate() {
        println!(
            "  {} stage {}: {}",
            dim("·"),
            i + 1,
            tendril_cluster::coordinator::spec_label(s)
        );
    }
    if max_diff == 0.0 {
        println!(
            "{} Bit-identical: splitting this model changes nothing.",
            ok_mark()
        );
    } else if max_diff < 1e-3 && agree == ids.len() {
        println!("{} Equivalent within floating-point noise.", ok_mark());
    } else {
        bail!("split execution differs from single-stage execution (max |Δ| {max_diff})");
    }
    Ok(())
}
