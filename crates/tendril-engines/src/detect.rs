//! Finding programs and Python packages on a machine.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Environment variable that points Tendril at an engine program
/// (`TENDRIL_LLAMA_SERVER=/opt/llama/bin/llama-server`).
pub fn env_name(program: &str) -> String {
    format!(
        "TENDRIL_{}",
        program
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            })
            .collect::<String>()
    )
}

/// A program on PATH (or in a few usual install places), honouring a
/// `TENDRIL_<NAME>` override.
pub fn find_program(names: &[&str]) -> Option<PathBuf> {
    for n in names {
        if let Ok(p) = std::env::var(env_name(n)) {
            let p = PathBuf::from(p);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    if let Some(h) = dirs::home_dir() {
        dirs.push(h.join(".local/bin"));
        dirs.push(h.join(".cargo/bin"));
    }
    dirs.push("/opt/homebrew/bin".into());
    dirs.push("/usr/local/bin".into());
    for n in names {
        for d in &dirs {
            for cand in [d.join(n), d.join(format!("{n}.exe"))] {
                if is_executable(&cand) {
                    return Some(cand);
                }
            }
        }
    }
    None
}

fn is_executable(p: &Path) -> bool {
    let Ok(m) = std::fs::metadata(p) else {
        return false;
    };
    if !m.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        m.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Run a command and return its stdout+stderr, or None if it can't run or
/// takes longer than `timeout`.
pub async fn run_capture(cmd: &[String], timeout: Duration) -> Option<String> {
    let (prog, args) = cmd.split_first()?;
    let fut = tokio::process::Command::new(prog)
        .args(args)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    let out = tokio::time::timeout(timeout, fut).await.ok()?.ok()?;
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    Some(s)
}

/// Like [`run_capture`] but only when the command exits successfully.
pub async fn run_ok(cmd: &[String], timeout: Duration) -> Option<String> {
    let (prog, args) = cmd.split_first()?;
    let fut = tokio::process::Command::new(prog)
        .args(args)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    let out = tokio::time::timeout(timeout, fut).await.ok()?.ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The Python interpreter engines written in Python run with
/// (`TENDRIL_PYTHON` overrides; an active virtualenv wins over the system one).
pub fn python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("TENDRIL_PYTHON") {
        return Some(PathBuf::from(p));
    }
    if let Ok(v) = std::env::var("VIRTUAL_ENV") {
        for c in ["bin/python", "Scripts/python.exe"] {
            let p = Path::new(&v).join(c);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    find_program(&["python3", "python"])
}

/// The version of a Python package importable by `python`, if any.
pub async fn python_package(python: &Path, module: &str) -> Option<String> {
    let code = format!(
        "import importlib.metadata as m, importlib.util as u, sys\n\
         if u.find_spec('{module}') is None: sys.exit(1)\n\
         try: print(m.version('{module}'))\n\
         except Exception: print('installed')"
    );
    run_ok(
        &[python.display().to_string(), "-c".into(), code],
        Duration::from_secs(20),
    )
    .await
}

/// Number of NVIDIA GPUs (`nvidia-smi -L`).
pub fn nvidia_gpus() -> u32 {
    let Ok(out) = std::process::Command::new("nvidia-smi").arg("-L").output() else {
        return 0;
    };
    if !out.status.success() {
        return 0;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.starts_with("GPU "))
        .count() as u32
}

/// First line that looks like a version in `text`.
pub fn first_version(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|l| l.chars().any(|c| c.is_ascii_digit()) && !l.is_empty())
        .map(|l| l.chars().take(80).collect())
}

/// A free TCP port on 127.0.0.1 (best effort: another process could take
/// it before the engine binds it).
pub fn free_port() -> std::io::Result<u16> {
    let l = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    Ok(l.local_addr()?.port())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_names() {
        assert_eq!(env_name("llama-server"), "TENDRIL_LLAMA_SERVER");
        assert_eq!(env_name("rpc-server"), "TENDRIL_RPC_SERVER");
    }

    #[tokio::test]
    async fn finds_sh() {
        if cfg!(unix) {
            assert!(find_program(&["sh"]).is_some());
            assert!(find_program(&["surely-not-a-program-xyz"]).is_none());
            let out = run_ok(
                &["sh".into(), "-c".into(), "echo hi".into()],
                Duration::from_secs(5),
            )
            .await;
            assert_eq!(out.as_deref(), Some("hi"));
        }
    }
}
