//! End-to-end: two models served from one pool of machines.

use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tendril_cluster::agent::{self, AgentOptions};
use tendril_cluster::coordinator::{Coordinator, ServeOptions, Status};
use tendril_cluster::http::{router, AppState};
use tendril_cluster::pool::Pool;
use tendril_core::Goal;
use tendril_engine::generate::{GenEvent, GenerateRequest, LocalModel, ModelFiles};
use tendril_engine::linear::WeightFormat;
use tendril_engine::sampler::SamplingParams;

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn opts(dir: &std::path::Path, name: &str, port: u16) -> ServeOptions {
    ServeOptions {
        model_dir: dir.to_path_buf(),
        model_name: name.into(),
        format: WeightFormat::Native,
        context: 512,
        concurrency: 2,
        goal: Goal::Balanced,
        safety: 0.05,
        control_port: port,
        data_port: 0,
        token: "POOL-TEST".into(),
        device: "cpu".into(),
        use_local: true,
        link: None,
        name: Some("coord".into()),
        // Split each model across both machines so the helper hosts a stage
        // of each: two sessions, two stages, one machine.
        min_stages: 2,
        max_memory: None,
        prefix_cache: true,
        kv_disk: tendril_core::Bytes::mib(16.0),
        speculate: true,
        draft_tokens: 4,
        recovery_timeout: Duration::from_secs(30),
    }
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

async fn wait_ready(c: &Coordinator) {
    for _ in 0..400 {
        if matches!(c.status(), Status::Ready { .. }) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("not ready: {:?}\n{:#?}", c.status(), c.history());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_models_share_the_machines_and_route_by_name() {
    let a_dir = tempfile::tempdir().unwrap();
    let b_dir = tempfile::tempdir().unwrap();
    tendril_engine::testing::write_tiny_llama(a_dir.path(), 4, 64, 3, candle_core::DType::F32)
        .unwrap();
    tendril_engine::testing::write_tiny_llama(b_dir.path(), 6, 64, 9, candle_core::DType::F32)
        .unwrap();
    let cache = tempfile::tempdir().unwrap();
    let port = free_port();
    let pool = Pool::start(vec![
        opts(a_dir.path(), "org/alpha", port),
        opts(b_dir.path(), "org/beta", port),
    ])
    .await
    .unwrap();

    // One `tendril join` serves every model of the pool.
    let agent = tokio::spawn(agent::run(
        AgentOptions {
            coordinator: format!("127.0.0.1:{port}"),
            token: "pool-test".into(),
            name: Some("helper".into()),
            device: "cpu".into(),
            data_port: 0,
            cache: cache.path().to_path_buf(),
            once: false,
            max_memory: None,
        },
        Arc::new(|m: &str, e| eprintln!("agent[{m}]: {e:?}")),
    ));
    for c in pool.models() {
        wait_ready(c).await;
        let nodes = c.nodes();
        assert_eq!(nodes.len(), 2, "{nodes:?}");
        assert!(nodes.iter().all(|n| n.role != "idle"), "{nodes:?}");
    }
    // Each model got a share of both machines.
    let shares = pool.shares();
    assert!(
        shares.iter().all(|s| s.placed && s.machines.len() == 2),
        "{shares:?}"
    );

    // HTTP: the `model` field picks the model.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http = listener.local_addr().unwrap();
    let app = router(AppState {
        pool: pool.clone(),
        join_command: String::new(),
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    let client = reqwest::Client::new();
    let base = format!("http://{http}");

    let models: Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["org/alpha", "org/beta"]);

    let prompt = pool.models()[0]
        .inner
        .tok
        .encode_prompt("hello pool")
        .unwrap();
    for (model, dir) in [("beta", b_dir.path()), ("org/alpha", a_dir.path())] {
        let r: Value = client
            .post(format!("{base}/v1/completions"))
            .json(
                &json!({"model": model, "prompt": "hello pool", "max_tokens": 24,
                          "temperature": 0, "ignore_eos": true}),
            )
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let want = reference(dir, &prompt, 24);
        assert_eq!(
            r["choices"][0]["text"].as_str().unwrap(),
            want,
            "{model}: {r}"
        );
        assert!(r["model"]
            .as_str()
            .unwrap()
            .ends_with(model.trim_start_matches("org/")));
    }

    // Unknown model: 404 naming the available ones.
    let r = client
        .post(format!("{base}/v1/completions"))
        .json(&json!({"model": "gamma", "prompt": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
    let body: Value = r.json().await.unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("org/alpha"));

    // Status lists both models with their shares.
    let st: Value = client
        .get(format!("{base}/api/status?model=beta"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(st["model"], "org/beta");
    assert_eq!(st["models"].as_array().unwrap().len(), 2);
    assert_eq!(st["models"][1]["state"], "ready");

    pool.shutdown().await;
    agent.abort();
}
