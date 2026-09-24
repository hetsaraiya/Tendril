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
    a.iter()
        .zip(b)
        .fold(0f64, |m, (x, y)| m.max((*x as f64 - y).abs()))
        / scale
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
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
                StageOutput::Nothing | StageOutput::AllLogits(_) => panic!("no logits"),
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
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
    let ids: Vec<u32> = reference["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let logits: Vec<Vec<f64>> = reference["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    let cfg = Arc::new(ModelConfig::from_file(&dir.join("config.json")).unwrap());
    let ws = WeightStore::open_dir(dir).unwrap();
    let bf16 = name.ends_with("-bf16");
    let tol = if bf16 { 0.08 } else { 2e-3 };
    let opts = LoadOptions {
        format: WeightFormat::Native,
        device: candle_core::Device::Cpu,
        dtype: candle_core::DType::F32,
    };

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
    println!(
        "{name:14} single-stage max rel err {worst:.2e}, argmax agreement {agree}/{}",
        single.len()
    );
    assert!(
        worst < tol,
        "{name}: logits differ from transformers by {worst}"
    );

    // Three-stage pipeline must match the single stage.
    let n = cfg.num_layers;
    let specs = [
        StageSpec {
            layer_start: 0,
            layer_end: 2,
            embed: true,
            head: false,
        },
        StageSpec {
            layer_start: 2,
            layer_end: n - 1,
            embed: false,
            head: false,
        },
        StageSpec {
            layer_start: n - 1,
            layer_end: n,
            embed: false,
            head: true,
        },
    ];
    let mut piped: Vec<Stage> = specs
        .iter()
        .map(|s| Stage::load(cfg.clone(), &ws, *s, &opts).unwrap())
        .collect();
    let split = run(&mut piped, &ids, 9);
    let mut diff = 0f32;
    for ((_, a), (_, b)) in single.iter().zip(&split) {
        for (x, y) in a.iter().zip(b) {
            diff = diff.max((x - y).abs());
        }
    }
    println!("{name:14} 3-stage vs 1-stage max abs diff {diff:.2e}");
    assert!(
        diff < 1e-4,
        "{name}: pipeline differs from single stage by {diff}"
    );
}

#[test]
fn matches_transformers() {
    let Ok(root) = std::env::var("TENDRIL_TEST_MODELS") else {
        eprintln!("TENDRIL_TEST_MODELS not set; skipping reference comparison");
        return;
    };
    let mut dirs: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("reference.json").exists())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty());
    for d in dirs {
        check_dir(&d);
    }
}

/// Batched decoding must equal one-at-a-time decoding.
#[test]
fn batch_equals_sequential() {
    let dir = tempfile::tempdir().unwrap();
    tendril_engine::testing::write_tiny_llama(dir.path(), 3, 64, 9, candle_core::DType::F32)
        .unwrap();
    let cfg = Arc::new(ModelConfig::from_file(&dir.path().join("config.json")).unwrap());
    let ws = WeightStore::open_dir(dir.path()).unwrap();
    let opts = LoadOptions {
        format: WeightFormat::Native,
        device: candle_core::Device::Cpu,
        dtype: candle_core::DType::F32,
    };
    let prompts: Vec<Vec<u32>> = vec![
        vec![5, 9, 22, 31],
        vec![7, 8],
        vec![40, 41, 42, 43, 44, 45, 46],
    ];
    let spec = StageSpec::whole(&cfg);
    // Sequential reference.
    let mut a = Stage::load(cfg.clone(), &ws, spec, &opts).unwrap();
    let mut want = Vec::new();
    for (s, p) in prompts.iter().enumerate() {
        let mut l = match a
            .forward(s as u64, 0, StageInput::Tokens(p.clone()), true)
            .unwrap()
        {
            StageOutput::Logits(l) => l,
            _ => panic!(),
        };
        let mut steps = vec![l.clone()];
        for i in 0..5 {
            let t = argmax(&l) as u32;
            l = match a
                .forward(s as u64, p.len() + i, StageInput::Tokens(vec![t]), true)
                .unwrap()
            {
                StageOutput::Logits(l) => l,
                _ => panic!(),
            };
            steps.push(l.clone());
        }
        want.push(steps);
    }
    // Batched: all prefills in one call, then all decode steps together.
    let mut b = Stage::load(cfg.clone(), &ws, spec, &opts).unwrap();
    let items = prompts
        .iter()
        .enumerate()
        .map(|(s, p)| tendril_engine::model::BatchItem {
            seq: s as u64,
            pos: 0,
            input: StageInput::Tokens(p.clone()),
            want_logits: true,
            all_logits: false,
        })
        .collect();
    let mut cur: Vec<Vec<f32>> = b
        .forward_batch(items)
        .into_iter()
        .map(|r| match r.unwrap() {
            StageOutput::Logits(l) => l,
            _ => panic!(),
        })
        .collect();
    for (s, c) in cur.iter().enumerate() {
        assert!(c.iter().zip(&want[s][0]).all(|(x, y)| (x - y).abs() < 1e-4));
    }
    for i in 0..5 {
        let items = cur
            .iter()
            .enumerate()
            .map(|(s, l)| tendril_engine::model::BatchItem {
                seq: s as u64,
                pos: prompts[s].len() + i,
                input: StageInput::Tokens(vec![argmax(l) as u32]),
                want_logits: true,
                all_logits: false,
            })
            .collect();
        cur = b
            .forward_batch(items)
            .into_iter()
            .map(|r| match r.unwrap() {
                StageOutput::Logits(l) => l,
                _ => panic!(),
            })
            .collect();
        for (s, c) in cur.iter().enumerate() {
            let d = c
                .iter()
                .zip(&want[s][i + 1])
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(d < 1e-4, "seq {s} step {i}: {d}");
        }
    }
    // A wrong position fails only that item.
    let r = b.forward_batch(vec![
        tendril_engine::model::BatchItem {
            seq: 0,
            pos: 999,
            input: StageInput::Tokens(vec![5]),
            want_logits: true,
            all_logits: false,
        },
        tendril_engine::model::BatchItem {
            seq: 1,
            pos: prompts[1].len() + 5,
            input: StageInput::Tokens(vec![5]),
            want_logits: true,
            all_logits: false,
        },
    ]);
    assert!(r[0].is_err() && r[1].is_ok());
}

/// Truncating to a shared prefix, or spilling to disk and restoring, must
/// continue exactly as if the prefix had been recomputed.
#[test]
fn prefix_reuse_and_spill_are_exact() {
    let dir = tempfile::tempdir().unwrap();
    tendril_engine::testing::write_tiny_llama(dir.path(), 3, 64, 11, candle_core::DType::F32)
        .unwrap();
    let cfg = Arc::new(ModelConfig::from_file(&dir.path().join("config.json")).unwrap());
    let ws = WeightStore::open_dir(dir.path()).unwrap();
    let opts = LoadOptions {
        format: WeightFormat::Native,
        device: candle_core::Device::Cpu,
        dtype: candle_core::DType::F32,
    };
    let mut s = Stage::load(cfg.clone(), &ws, StageSpec::whole(&cfg), &opts).unwrap();
    let logits = |o: StageOutput| match o {
        StageOutput::Logits(l) => l,
        _ => panic!(),
    };
    // Reference: fresh prefill of the second turn.
    let turn2: Vec<u32> = (5..60).collect();
    let want = logits(
        s.forward(1, 0, StageInput::Tokens(turn2.clone()), true)
            .unwrap(),
    );
    // Turn 1 shares the first 40 tokens, then diverges.
    let mut turn1: Vec<u32> = (5..45).collect();
    turn1.extend([200, 201, 202]);
    s.forward(2, 0, StageInput::Tokens(turn1), true).unwrap();
    s.truncate(2, 40).unwrap();
    let got = logits(
        s.forward(2, 40, StageInput::Tokens(turn2[40..].to_vec()), true)
            .unwrap(),
    );
    assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-5));
    // Spill and restore.
    let f = dir.path().join("kv/2.kv");
    s.spill(2, &f).unwrap();
    assert!(!s.has_seq(2));
    s.restore(2, &f).unwrap();
    s.truncate(2, 40).unwrap();
    let again = logits(
        s.forward(2, 40, StageInput::Tokens(turn2[40..].to_vec()), true)
            .unwrap(),
    );
    assert!(again.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-5));
    assert!(s.truncate(2, 1000).is_err());
}
