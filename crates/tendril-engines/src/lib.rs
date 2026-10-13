//! Inference engines Tendril runs models with.
//!
//! Tendril plans, moves files, secures the network and serves one API; the
//! engines in [`engines`] run the models. Each one is a type implementing
//! [`Engine`]: what it needs installed, what it runs, whether and how it
//! spans machines, how its traffic is protected, and how to start it.

#![allow(clippy::too_many_arguments)]

pub mod detect;
pub mod engine;
pub mod engines;
pub mod hf;
pub mod spec;
pub mod util;

pub use engine::*;

use std::sync::Arc;

/// Every engine Tendril knows, in the order the dashboard lists them.
pub fn all() -> Vec<Arc<dyn Engine>> {
    engines::all()
}

/// The engine with this id ("llama.cpp", "vllm", …).
pub fn get(id: &str) -> Option<Arc<dyn Engine>> {
    let id = id.trim().to_ascii_lowercase();
    all().into_iter().find(|e| {
        let i = e.info();
        i.id == id
            || i.name.to_ascii_lowercase() == id
            || i.id.replace('.', "") == id.replace('.', "")
    })
}

/// Detect every engine on this machine.
pub async fn detect_all() -> Vec<Detection> {
    let host = tokio::task::spawn_blocking(HostFacts::this_machine)
        .await
        .unwrap_or(HostFacts {
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            nvidia_gpus: 0,
            apple_silicon: false,
        });
    let futs = all().into_iter().map(|e| {
        let host = host.clone();
        async move { e.detect(&host).await }
    });
    futures::future::join_all(futs).await
}
