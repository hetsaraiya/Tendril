//! This machine as the planner sees it.

use tendril_core::hardware::{detect_local, NodeProfile};
use tendril_core::Bytes;

/// Detect this machine (with its saved benchmark), named and capped as asked.
pub fn this_machine(name: Option<&str>, max_memory: Option<Bytes>) -> NodeProfile {
    let mut p = detect_local(true);
    if let Some(n) = name {
        p.name = n.to_string();
    }
    apply_memory_cap(&mut p, max_memory);
    p
}

/// Respect a user's `--max-memory` cap.
pub fn apply_memory_cap(p: &mut NodeProfile, cap: Option<Bytes>) {
    if let Some(cap) = cap {
        if cap < p.usable_memory {
            p.usable_memory = cap;
            p.usable_reason = format!("limited to {cap} by --max-memory");
        }
    }
}

/// Local wall-clock time for event logs.
pub fn clock() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

/// Where this machine keeps downloaded models (`TENDRIL_CACHE` overrides).
pub fn default_cache() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("TENDRIL_CACHE") {
        return p.into();
    }
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("tendril")
}
