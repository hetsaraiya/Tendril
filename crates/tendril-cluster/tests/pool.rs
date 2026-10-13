//! The pool's rules, with stand-in engine programs: engines whose traffic
//! Tendril can't encrypt only span machines after the user confirms a
//! trusted network, and unloading stops the processes.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tendril_cluster::agent::{self, AgentOptions};
use tendril_cluster::pool::{LoadRequest, NeedsTrust, Pool, PoolOptions, Status};

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn trusted_network_is_required_to_span_machines() {
    use std::os::unix::fs::PermissionsExt;
    // A fake `exo` that just runs (it never answers HTTP, so it stays loading).
    let dir = tempfile::tempdir().unwrap();
    let fake = dir.path().join("exo");
    let pidfile = dir.path().join("pids");
    std::fs::write(
        &fake,
        format!(
            "#!/bin/sh\necho $$ >> {}\nexec sleep 600\n",
            pidfile.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("TENDRIL_EXO", &fake);
    // A stand-in HuggingFace that lists the model's files (for its size).
    let hf = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    std::env::set_var(
        "TENDRIL_HF_ENDPOINT",
        format!("http://{}", hf.local_addr().unwrap()),
    );
    let app = axum::Router::new().route(
        "/api/models/{org}/{name}/tree/{rev}",
        axum::routing::get(|| async {
            axum::Json(serde_json::json!([{"type": "file", "path": "model.safetensors", "size": 300_000_000u64}]))
        }),
    );
    tokio::spawn(async move { axum::serve(hf, app).await.unwrap() });

    let token = "POOL-TEST".to_string();
    let pool = Pool::start(PoolOptions {
        control_port: 0,
        token: token.clone(),
        name: "t".into(),
        default_context: 512,
        default_concurrency: 1,
    })
    .await
    .unwrap();
    for (name, local) in [("a", true), ("b", false)] {
        let opts = AgentOptions {
            server: format!("127.0.0.1:{}", pool.control_port),
            token: token.clone(),
            name: Some(name.into()),
            tunnel_port: 0,
            cache: dir.path().join(name),
            once: true,
            max_memory: None,
            local,
        };
        tokio::spawn(agent::run(opts, Arc::new(|_| {})));
    }
    let t0 = Instant::now();
    while pool.machine_count() < 2 {
        assert!(t0.elapsed() < Duration::from_secs(60));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let mut req = LoadRequest {
        engine: "exo".into(),
        model: "mlx-community/Tiny-4bit".into(),
        ..Default::default()
    };
    // A model-card name has no known size: refused rather than unreserved.
    let card = LoadRequest {
        model: "llama-3.2-1b".into(),
        ..req.clone()
    };
    assert!(pool.plan(&card).await.is_err());
    let plan = pool.plan(&req).await.unwrap();
    assert!(plan.fits, "{:?}", plan.reason);
    assert!(plan.weight_bytes >= 300_000_000);
    assert_eq!(plan.stages.len(), 2);
    assert!(plan.needs_trusted_network);
    assert!(plan.risk.is_some());

    // Without consent: refused, nothing starts.
    let e = pool.load(&req).await.unwrap_err();
    assert!(e.is::<NeedsTrust>(), "{e:#}");
    assert!(pool.deployments().is_empty());

    // A stale plan is refused too.
    req.trusted_network = true;
    req.expect_version = Some(plan.pool_version + 1);
    assert!(pool.load(&req).await.is_err());

    // With consent: both machines start exo.
    req.expect_version = Some(plan.pool_version);
    let name = pool.load(&req).await.unwrap();
    let t0 = Instant::now();
    loop {
        let n = std::fs::read_to_string(&pidfile)
            .map(|s| s.lines().count())
            .unwrap_or(0);
        if n >= 2 {
            break;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "exo didn't start on both machines"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(pool.deployments()[0].status(), Status::Loading);

    // Unloading stops them.
    pool.unload(&name).unwrap();
    let pids: Vec<i32> = std::fs::read_to_string(&pidfile)
        .unwrap()
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let t0 = Instant::now();
    while pids.iter().any(|p| unsafe { libc::kill(*p, 0) } == 0) {
        assert!(
            t0.elapsed() < Duration::from_secs(15),
            "engine processes still running"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    pool.shutdown().await;
}
