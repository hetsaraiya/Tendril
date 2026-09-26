//! End-to-end: a coordinator and agents on localhost serving a tiny model.

use std::sync::Arc;
use std::time::Duration;
use tendril_cluster::agent::{self, AgentOptions};
use tendril_cluster::coordinator::{Coordinator, GenOut, GenRequest, ServeOptions, Status};
use tendril_core::Goal;
use tendril_engine::generate::{GenEvent, GenerateRequest, LocalModel, ModelFiles};
use tendril_engine::linear::WeightFormat;
use tendril_engine::sampler::SamplingParams;
use tendril_engine::tokenizer::ChatMessage;

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn serve_opts(dir: &std::path::Path, port: u16, local: bool, min_stages: usize) -> ServeOptions {
    ServeOptions {
        model_dir: dir.to_path_buf(),
        model_name: "tiny".into(),
        format: WeightFormat::Native,
        context: 512,
        concurrency: 4,
        goal: Goal::Balanced,
        safety: 0.05,
        control_port: port,
        data_port: 0,
        token: "TEST-TOKE-N123-4567".into(),
        device: "cpu".into(),
        use_local: local,
        link: None,
        name: Some("coord".into()),
        min_stages,
        max_memory: None,
        prefix_cache: true,
        kv_disk: tendril_core::Bytes::mib(64.0),
        speculate: true,
        draft_tokens: 6,
        recovery_timeout: Duration::from_secs(30),
        remote: None,
    }
}

fn spawn_agent(
    port: u16,
    name: &str,
    cache: &std::path::Path,
) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    let opts = AgentOptions {
        coordinator: format!("127.0.0.1:{port}"),
        token: "test-toke-n123 4567".into(), // case/format-insensitive
        name: Some(name.into()),
        device: "cpu".into(),
        data_port: 0,
        cache: cache.to_path_buf(),
        once: false,
        max_memory: None,
    };
    tokio::spawn(agent::run(
        opts,
        Arc::new(|m: &str, e| eprintln!("agent[{m}]: {e:?}")),
    ))
}

async fn wait_ready(c: &Coordinator) {
    for _ in 0..300 {
        if matches!(c.status(), Status::Ready { .. }) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("not ready: {:?}\n{:#?}", c.status(), c.history());
}

async fn collect(c: &Coordinator, prompt: Vec<u32>, max: usize) -> String {
    let mut rx = c
        .generate(GenRequest {
            prompt,
            params: SamplingParams::greedy(),
            max_tokens: max,
            stop: vec![],
            ignore_eos: false,
        })
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let mut s = String::new();
    while let Some(o) = rx.recv().await {
        match o {
            GenOut::Text(t) => s.push_str(&t),
            GenOut::Done { .. } => break,
            GenOut::Error(e) => panic!("{e}"),
        }
    }
    s
}

fn reference(dir: &std::path::Path, prompt: &[u32], max: usize) -> String {
    let mut m = LocalModel::load(
        ModelFiles::new(dir).unwrap(),
        candle_core::Device::Cpu,
        WeightFormat::Native,
        None,
    )
    .unwrap();
    let mut s = String::new();
    m.generate(
        GenerateRequest {
            prompt: prompt.to_vec(),
            params: SamplingParams::greedy(),
            max_tokens: max,
            stop: vec![],
            ignore_eos: false,
        },
        |e| {
            if let GenEvent::Text(t) = e {
                s.push_str(t);
            }
            true
        },
    )
    .unwrap();
    s
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_plus_remote_pipeline_matches_single_machine() {
    let dir = tempfile::tempdir().unwrap();
    tendril_engine::testing::write_tiny_llama(dir.path(), 4, 64, 3, candle_core::DType::F32)
        .unwrap();
    let cache = tempfile::tempdir().unwrap();
    let port = free_port();
    let c = Coordinator::start(serve_opts(dir.path(), port, true, 2))
        .await
        .unwrap();
    let a = spawn_agent(port, "helper", cache.path());
    wait_ready(&c).await;
    let nodes = c.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(nodes.iter().all(|n| n.role != "idle"), "{nodes:?}");
    let prompt = c
        .inner
        .tok
        .encode_chat(&[ChatMessage::new("user", "hello there")])
        .unwrap();
    let want = reference(dir.path(), &prompt, 24);
    // Several concurrent requests exercise pipelining and per-sequence KV.
    let mut handles = vec![];
    for _ in 0..3 {
        let c2 = c.clone();
        let p = prompt.clone();
        handles.push(tokio::spawn(async move { collect(&c2, p, 24).await }));
    }
    for h in handles {
        assert_eq!(h.await.unwrap(), want);
    }
    assert_eq!(c.metrics().requests, 3);

    // The helper leaves: requests are refused clearly, not hung.
    a.abort();
    for _ in 0..100 {
        if !matches!(c.status(), Status::Ready { .. }) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let r = c
        .generate(GenRequest {
            prompt: prompt.clone(),
            params: SamplingParams::greedy(),
            max_tokens: 4,
            stop: vec![],
            ignore_eos: false,
        })
        .await;
    assert!(
        r.is_err(),
        "should refuse while a required machine is missing"
    );

    // It comes back (with its weights cached) and serving resumes.
    let _a2 = spawn_agent(port, "helper", cache.path());
    wait_ready(&c).await;
    assert_eq!(collect(&c, prompt.clone(), 24).await, want);
    assert!(
        c.history().iter().any(|e| e.text.contains("already has")),
        "second load should hit the shard cache"
    );
    c.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_remote_stages() {
    let dir = tempfile::tempdir().unwrap();
    tendril_engine::testing::write_tiny_llama(dir.path(), 6, 64, 5, candle_core::DType::BF16)
        .unwrap();
    let port = free_port();
    let c = Coordinator::start(serve_opts(dir.path(), port, false, 3))
        .await
        .unwrap();
    let caches: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
    let _agents: Vec<_> = (0..3)
        .map(|i| spawn_agent(port, &format!("w{i}"), caches[i].path()))
        .collect();
    wait_ready(&c).await;
    let prompt: Vec<u32> = (0..300).map(|i| 5 + (i * 13 % 250) as u32).collect(); // multi-chunk prefill
    let want = reference(dir.path(), &prompt, 20);
    assert_eq!(collect(&c, prompt, 20).await, want);
    // The repetitive prompt makes prompt-lookup drafts verify in bulk, and the
    // output above still matched the reference exactly.
    let m = c.metrics();
    assert!(m.spec_steps > 0, "{m:?}");
    eprintln!(
        "speculation: {} passes, {}/{} drafted tokens accepted",
        m.spec_steps, m.accepted_tokens, m.drafted_tokens
    );
    // Per-stage timings travel with every plain token.
    let (plan, tel) = c.telemetry().await.expect("running pipeline");
    assert!(tel.samples >= 1 || m.spec_steps > 0, "{tel:?}");
    if tel.samples > 0 {
        assert_eq!(tel.compute_ms.len(), plan.stages.len());
        assert!(tel.compute_ms.iter().all(|&m| m > 0.0));
        assert!(tel.step_ms >= tel.compute_ms.iter().sum::<f64>() * 0.5);
    }
    c.shutdown().await;
}

async fn collect_with_timing(
    c: &Coordinator,
    prompt: Vec<u32>,
    max: usize,
) -> (String, tendril_engine::generate::Timing) {
    let mut rx = c
        .generate(GenRequest {
            prompt,
            params: SamplingParams::greedy(),
            max_tokens: max,
            stop: vec![],
            ignore_eos: true,
        })
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let mut s = String::new();
    while let Some(o) = rx.recv().await {
        match o {
            GenOut::Text(t) => s.push_str(&t),
            GenOut::Done { timing, .. } => return (s, timing),
            GenOut::Error(e) => panic!("{e}"),
        }
    }
    panic!("no Done")
}

fn reference_ignore_eos(dir: &std::path::Path, prompt: &[u32], max: usize) -> String {
    let mut m = LocalModel::load(
        ModelFiles::new(dir).unwrap(),
        candle_core::Device::Cpu,
        WeightFormat::Native,
        None,
    )
    .unwrap();
    let mut s = String::new();
    m.generate(
        GenerateRequest {
            prompt: prompt.to_vec(),
            params: SamplingParams::greedy(),
            max_tokens: max,
            stop: vec![],
            ignore_eos: true,
        },
        |e| {
            if let GenEvent::Text(t) = e {
                s.push_str(t);
            }
            true
        },
    )
    .unwrap();
    s
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prefix_cache_reuses_spills_and_restores() {
    let dir = tempfile::tempdir().unwrap();
    tendril_engine::testing::write_tiny_llama(dir.path(), 4, 64, 21, candle_core::DType::F32)
        .unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::env::set_var("TENDRIL_CACHE", cache.path());
    let port = free_port();
    let mut opts = serve_opts(dir.path(), port, true, 2);
    opts.concurrency = 1; // one KV slot: parking a second conversation forces a spill
    let c = Coordinator::start(opts).await.unwrap();
    let _a = spawn_agent(port, "helper", cache.path());
    wait_ready(&c).await;

    let turn1: Vec<u32> = (0..60).map(|i| 5 + (i * 7 % 200) as u32).collect();
    let (_, t1) = collect_with_timing(&c, turn1.clone(), 8).await;
    assert_eq!(t1.cached_tokens, 0);
    // Turn 2 extends turn 1: its first 60 tokens come from the cache.
    let mut turn2 = turn1.clone();
    turn2.extend([40, 41, 42, 43, 44, 45]);
    let (text2, t2) = collect_with_timing(&c, turn2.clone(), 10).await;
    assert!(t2.cached_tokens >= 60, "cached {}", t2.cached_tokens);
    let fresh = reference_ignore_eos(dir.path(), &turn2, 10);
    assert_eq!(
        text2, fresh,
        "reusing the prefix must not change the output"
    );
    // An unrelated request needs the only memory slot: turn 2 spills to disk.
    let other: Vec<u32> = (0..30).map(|i| 200 + i as u32).collect();
    collect_with_timing(&c, other, 4).await;
    assert!(c.metrics().kv_spills >= 1, "{:?}", c.metrics());
    // Turn 3 restores it from disk and continues.
    let mut turn3 = turn2.clone();
    turn3.extend([50, 51, 52]);
    let (text3, t3) = collect_with_timing(&c, turn3.clone(), 10).await;
    assert!(t3.cached_tokens >= 66, "cached {}", t3.cached_tokens);
    assert!(c.metrics().kv_restores >= 1);
    let fresh3 = reference_ignore_eos(dir.path(), &turn3, 10);
    assert_eq!(
        text3, fresh3,
        "restoring from disk must not change the output"
    );
    c.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn speculation_is_exact_and_accepts_repetitive_text() {
    let dir = tempfile::tempdir().unwrap();
    tendril_engine::testing::write_tiny_llama(dir.path(), 4, 64, 3, candle_core::DType::F32)
        .unwrap();
    let cache = tempfile::tempdir().unwrap();
    let port = free_port();
    let c = Coordinator::start(serve_opts(dir.path(), port, true, 2))
        .await
        .unwrap();
    let _a = spawn_agent(port, "helper", cache.path());
    wait_ready(&c).await;
    // Greedy decoding of a small random model settles into a cycle: prompt
    // lookup should draft it and the pipeline should accept those drafts.
    let prompt: Vec<u32> = (0..24).map(|i| 5 + (i * 11 % 90) as u32).collect();
    let (text, _) = collect_with_timing(&c, prompt.clone(), 160).await;
    let want = reference_ignore_eos(dir.path(), &prompt, 160);
    assert_eq!(
        text, want,
        "speculative output must equal plain greedy output"
    );
    let m = c.metrics();
    eprintln!(
        "speculation: {} passes, {}/{} drafted tokens accepted",
        m.spec_steps, m.accepted_tokens, m.drafted_tokens
    );
    assert!(m.accepted_tokens > 0, "{m:?}");
    c.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
// The rejoin task hands back the new agent's handle so it stays alive.
#[allow(clippy::async_yields_async)]
async fn request_survives_a_machine_leaving_mid_generation() {
    let dir = tempfile::tempdir().unwrap();
    tendril_engine::testing::write_tiny_llama(dir.path(), 6, 128, 13, candle_core::DType::F32)
        .unwrap();
    let cache = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut opts = serve_opts(dir.path(), port, true, 2);
    opts.speculate = false; // keep steps deterministic for the comparison
    let c = Coordinator::start(opts).await.unwrap();
    let a = spawn_agent(port, "helper", cache.path());
    wait_ready(&c).await;
    let prompt: Vec<u32> = (0..40).map(|i| 5 + (i * 17 % 180) as u32).collect();
    let want = reference_ignore_eos(dir.path(), &prompt, 300);
    let mut rx = c
        .generate(GenRequest {
            prompt: prompt.clone(),
            params: SamplingParams::greedy(),
            max_tokens: 300,
            stop: vec![],
            ignore_eos: true,
        })
        .await
        .unwrap();
    let mut text = String::new();
    let mut chunks = 0;
    let mut agent = Some(a);
    let mut rejoined = None;
    let done = loop {
        match rx.recv().await {
            Some(GenOut::Text(t)) => {
                text.push_str(&t);
                chunks += 1;
                if chunks == 25 {
                    // The helper disappears mid-generation...
                    agent.take().unwrap().abort();
                    let port2 = port;
                    let cache2 = cache.path().to_path_buf();
                    // ...and comes back a moment later (its shard is cached).
                    rejoined = Some(tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(800)).await;
                        spawn_agent(port2, "helper", &cache2)
                    }));
                }
            }
            Some(GenOut::Done { timing, .. }) => break timing,
            Some(GenOut::Error(e)) => panic!(
                "request failed instead of recovering: {e}\n{:#?}",
                c.history()
            ),
            None => panic!("stream ended"),
        }
    };
    assert!(
        rejoined.is_some(),
        "the machine should have been removed mid-stream"
    );
    assert_eq!(done.completion_tokens, 300);
    assert_eq!(c.metrics().recoveries, 1, "{:#?}", c.history());
    assert_eq!(text, want, "no lost or duplicated text across the recovery");
    c.shutdown().await;
}
