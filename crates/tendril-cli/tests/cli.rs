//! End-to-end checks of the `tendril` binary.

use std::process::Command;

fn tendril(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_tendril"))
        .args(args)
        .env("NO_COLOR", "1")
        .output()
        .expect("run tendril");
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), s)
}

#[test]
fn plans_two_macs() {
    let (ok, out) = tendril(&[
        "plan",
        "gemma-2-9b",
        "--node",
        "a=m4:16",
        "--node",
        "b=m5:16",
        "--quantize",
        "q8_0",
        "--link",
        "tb",
    ]);
    assert!(ok, "{out}");
    assert!(out.contains("Runs across 2 machines"), "{out}");
    assert!(out.contains("final norm + LM head"), "{out}");
}

#[test]
fn explains_infeasible() {
    let (ok, out) = tendril(&["plan", "llama-3.1-70b", "--node", "a=m4:16"]);
    assert!(ok, "{out}");
    assert!(out.contains("does not fit"), "{out}");
    assert!(out.contains("How to make it fit"), "{out}");
}

#[test]
fn json_is_parseable() {
    let (ok, out) = tendril(&["plan", "qwen2.5-0.5b", "--node", "a=m4:16", "--json"]);
    assert!(ok, "{out}");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(v["result"]["selected"]["stages"].as_array().unwrap().len() == 1);
}

#[test]
fn friendly_errors() {
    let (ok, out) = tendril(&["plan", "lama-3.1-8b"]);
    assert!(!ok);
    assert!(out.contains("Did you mean"), "{out}");
    let (ok, out) = tendril(&["plan", "llama-3.1-8b", "--node", "x=nonsense"]);
    assert!(!ok);
    assert!(out.contains("unknown hardware"), "{out}");
}

#[test]
fn other_commands_run() {
    for args in [
        vec!["models"],
        vec!["hardware"],
        vec!["inspect", "gemma-3-4b"],
        vec!["fit", "qwen3-8b", "--node", "a=m4-pro:24"],
        vec!["node"],
    ] {
        let (ok, out) = tendril(&args);
        assert!(ok, "{args:?}: {out}");
    }
}
