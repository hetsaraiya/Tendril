//! A real engine through Tendril: llama.cpp split across two machines (two
//! agents here), its RPC traffic carried by Tendril's encrypted tunnels.
//!
//! Needs llama.cpp built with RPC; skipped unless `TENDRIL_E2E_GGUF` points
//! at a small GGUF and `llama-server` / `ggml-rpc-server` are on PATH (or set
//! with TENDRIL_LLAMA_SERVER / TENDRIL_GGML_RPC_SERVER).

use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tendril_cluster::agent::{self, AgentOptions};
use tendril_cluster::http::{router, AppState};
use tendril_cluster::pool::{LoadRequest, Pool, PoolOptions, Status};

fn gguf() -> Option<String> {
    let p = std::env::var("TENDRIL_E2E_GGUF").ok()?;
    if tendril_engines::detect::find_program(&["llama-server"]).is_none()
        || tendril_engines::detect::find_program(&["ggml-rpc-server", "rpc-server"]).is_none()
    {
        eprintln!("skipping: llama-server / ggml-rpc-server not found");
        return None;
    }
    Some(p)
}

async fn wait_ready(pool: &Pool, name: &str, secs: u64) {
    let t0 = Instant::now();
    loop {
        let d = pool
            .deployments()
            .into_iter()
            .find(|d| d.name == name)
            .expect("deployment");
        match d.status() {
            Status::Ready => return,
            Status::Failed => panic!("deployment failed: {:#?}", d.view()),
            _ => {}
        }
        assert!(
            t0.elapsed() < Duration::from_secs(secs),
            "not ready: {:#?}",
            d.view()
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn chat(http: &reqwest::Client, base: &str, model: &str) -> String {
    let r: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": model,
            "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 16, "temperature": 0, "seed": 1
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    r["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("no answer: {r}"))
        .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn llama_cpp_split_over_tunnels_matches_one_machine() {
    let Some(model) = gguf() else {
        eprintln!("skipping: set TENDRIL_E2E_GGUF to run");
        return;
    };
    let token = "E2E-TEST-TOKEN".to_string();
    let pool = Pool::start(PoolOptions {
        control_port: 0,
        token: token.clone(),
        name: "e2e".into(),
        default_context: 512,
        default_concurrency: 1,
    })
    .await
    .unwrap();
    let cache = tempfile::tempdir().unwrap();
    for (name, local) in [("head", true), ("worker", false)] {
        let opts = AgentOptions {
            server: format!("127.0.0.1:{}", pool.control_port),
            token: token.clone(),
            name: Some(name.into()),
            tunnel_port: 0,
            cache: cache.path().join(name),
            once: true,
            max_memory: None,
            local,
        };
        tokio::spawn(agent::run(opts, Arc::new(|_| {})));
    }
    let t0 = Instant::now();
    while pool.machine_count() < 2 {
        assert!(t0.elapsed() < Duration::from_secs(60), "agents didn't join");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let app = router(AppState::new(pool.clone(), String::new()));
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    let http = reqwest::Client::new();

    // One machine first: the reference answer.
    let req = LoadRequest {
        engine: "llama.cpp".into(),
        model: model.clone(),
        machines: Some(vec!["head".into()]),
        ..Default::default()
    };
    let name = pool.load(&req).await.unwrap();
    wait_ready(&pool, &name, 120).await;
    let alone = chat(&http, &base, &name).await;
    pool.unload(&name).unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Now split across both: the plan says so before anything starts.
    let req = LoadRequest {
        engine: "llama.cpp".into(),
        model,
        min_machines: Some(2),
        ..Default::default()
    };
    let plan = pool.plan(&req).await.unwrap();
    assert!(plan.fits, "{:?}", plan.reason);
    assert_eq!(plan.stages.len(), 2, "{:#?}", plan.stages);
    assert!(
        !plan.needs_trusted_network,
        "llama.cpp traffic is Tendril-encrypted"
    );
    let name = pool.load(&req).await.unwrap();
    wait_ready(&pool, &name, 180).await;
    let view = pool.deployments()[0].view();
    assert_eq!(view.roles.len(), 2);
    let rpc = view.roles.iter().find(|r| r.id == "rpc-1").unwrap();
    assert_eq!(rpc.machine_name, "worker");
    let split = chat(&http, &base, &name).await;
    // llama-server really put layers on the RPC device (via the tunnel's local end).
    let cmd = &view
        .plan
        .commands
        .iter()
        .find(|(m, _)| m == "head")
        .unwrap()
        .1;
    assert!(cmd.contains("--rpc {peer:rpc-1}"), "{cmd}");
    assert_eq!(
        split, alone,
        "the split model answers exactly like one machine"
    );

    // The API lists it; unloading stops the engines on both machines.
    let models: Value = http
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], json!(name));
    pool.unload(&name).unwrap();
    pool.shutdown().await;
}
