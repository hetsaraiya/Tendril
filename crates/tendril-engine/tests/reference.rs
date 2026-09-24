//! Compare Tendril's engine against HuggingFace transformers.
//!
//! Generate fixtures with `python tools/make_test_models.py DIR` and run
//! `TENDRIL_TEST_MODELS=DIR cargo test -p tendril-engine --test reference`.
//! Skipped when the variable is not set.

use std::path::Path;
use std::sync::Arc;
use tendril_engine::config::ModelConfig;
use tendril_engine::linear::WeightFormat;
use tendril_engine::model::{LoadOptions, Stage, StageInput, StageOutput, StageSpec};
use tendril_engine::weights::WeightStore;

fn max_rel(a: &[f32], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-6);
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - y).abs())) / scale
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

/// Run a list of stages as a pipeline over `ids` (prefill `split` tokens,
/// then one token at a time); return logits for positions split-1.. .
fn run(stages: &mut [Stage], ids: &[u32], split: usize) -> Vec<(usize, Vec<f32>)> {
    let mut out = Vec::new();
    let mut pos = 0;
    let feed = |stages: &mut [Stage], toks: &[u32], pos: usize| -> Vec<f32> {
        let mut input = StageInput::Tokens(toks.to_vec());
        for s in stages.iter_mut() {
            match s.forward(7, pos, input, true).unwrap() {
                StageOutput::Hidden(h) => input = StageInput::Hidden(h),
                StageOutput::Logits(l) => return l,
                StageOutput::Nothing => panic!("no logits"),
            }
        }
        unreachable!()
    };
    let l = feed(stages, &ids[..split], pos);
    pos += split;
    out.push((split - 1, l));
    for i in split..ids.len() {
        let l = feed(stages, &ids[i..i + 1], pos);
        pos += 1;
        out.push((i, l));
    }
    out
}

fn check_dir(dir: &Path) {
    let name = dir.file_name().unwrap().to_string_lossy().to_string();
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
    let ids: Vec<u32> = reference["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let logits: Vec<Vec<f64>> = reference["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())
        .collect();
    let cfg = Arc::new(ModelConfig::from_file(&dir.join("config.json")).unwrap());
    let ws = WeightStore::open_dir(dir).unwrap();
    let bf16 = name.ends_with("-bf16");
    let tol = if bf16 { 0.08 } else { 2e-3 };
    let opts = LoadOptions { format: WeightFormat::Native, device: candle_core::Device::Cpu, dtype: candle_core::DType::F32 };

    // Whole model on one stage.
    let mut whole = vec![Stage::load(cfg.clone(), &ws, StageSpec::whole(&cfg), &opts).unwrap()];
    let single = run(&mut whole, &ids, 9);
    let mut worst = 0f64;
    let mut agree = 0;
    for (p, l) in &single {
        worst = worst.max(max_rel(l, &logits[*p]));
        let r: Vec<f32> = logits[*p].iter().map(|x| *x as f32).collect();
        agree += usize::from(argmax(l) == argmax(&r));
    }
    println!("{name:14} single-stage max rel err {worst:.2e}, argmax agreement {agree}/{}", single.len());
    assert!(worst < tol, "{name}: logits differ from transformers by {worst}");

    // Three-stage pipeline must match the single stage.
    let n = cfg.num_layers;
    let specs = [
        StageSpec { layer_start: 0, layer_end: 2, embed: true, head: false },
        StageSpec { layer_start: 2, layer_end: n - 1, embed: false, head: false },
        StageSpec { layer_start: n - 1, layer_end: n, embed: false, head: true },
    ];
    let mut piped: Vec<Stage> = specs.iter().map(|s| Stage::load(cfg.clone(), &ws, *s, &opts).unwrap()).collect();
    let split = run(&mut piped, &ids, 9);
    let mut diff = 0f32;
    for ((_, a), (_, b)) in single.iter().zip(&split) {
        for (x, y) in a.iter().zip(b) {
            diff = diff.max((x - y).abs());
        }
    }
    println!("{name:14} 3-stage vs 1-stage max abs diff {diff:.2e}");
    assert!(diff < 1e-4, "{name}: pipeline differs from single stage by {diff}");
}

#[test]
fn matches_transformers() {
    let Ok(root) = std::env::var("TENDRIL_TEST_MODELS") else {
        eprintln!("TENDRIL_TEST_MODELS not set; skipping reference comparison");
        return;
    };
    let mut dirs: Vec<_> = std::fs::read_dir(root).unwrap().flatten().map(|e| e.path()).filter(|p| p.join("reference.json").exists()).collect();
    dirs.sort();
    assert!(!dirs.is_empty());
    for d in dirs {
        check_dir(&d);
    }
}
