//! Small helpers shared by the engines.

use crate::engine::*;
use tendril_core::Bytes;

/// A launch with the usual defaults (localhost, HTTP health check).
pub fn launch(engine: &str, program: &str, args: &[&str]) -> Launch {
    Launch {
        engine: engine.into(),
        program: program.into(),
        args: args.iter().map(|s| s.to_string()).collect(),
        env: vec![],
        cwd: None,
        ports: vec![],
        files: vec![],
        peers: vec![],
        ready: Ready::Http {
            path: "/health".into(),
        },
        ready_timeout: 1800,
        lan: false,
        extra: serde_json::Value::Null,
    }
}

impl Launch {
    pub fn arg(mut self, a: impl Into<String>) -> Launch {
        self.args.push(a.into());
        self
    }

    pub fn args<I: IntoIterator<Item = S>, S: Into<String>>(mut self, a: I) -> Launch {
        self.args.extend(a.into_iter().map(Into::into));
        self
    }

    pub fn env(mut self, k: &str, v: impl Into<String>) -> Launch {
        self.env.push((k.into(), v.into()));
        self
    }

    pub fn ready(mut self, r: Ready) -> Launch {
        self.ready = r;
        self
    }
}

/// One role on one machine.
pub fn role(id: &str, machine: &str, serves_api: bool, launch: Launch) -> Role {
    Role {
        id: id.into(),
        machine: machine.into(),
        serves_api,
        after: vec![],
        after_started: vec![],
        launch,
    }
}

/// A one-stage placement's machine, or an error naming the engine.
pub fn single<'a>(engine: &str, placement: &'a Placement) -> anyhow::Result<&'a Stage> {
    match placement.stages.as_slice() {
        [s] => Ok(s),
        [] => anyhow::bail!("{engine}: nothing to run"),
        _ => anyhow::bail!("{engine} runs a model on one machine only"),
    }
}

/// A detection for an engine that isn't installed.
pub fn not_installed(info: &EngineInfo, host: &HostFacts) -> Detection {
    Detection {
        engine: info.id.clone(),
        ready: false,
        missing: info
            .requirements
            .iter()
            .map(|r| r.missing(&host.os))
            .collect(),
        ..Default::default()
    }
}

/// Accelerators a Python/GPU engine sees on this machine.
pub fn host_accelerators(host: &HostFacts) -> (Vec<String>, u32) {
    if host.nvidia_gpus > 0 {
        (vec!["CUDA".into()], host.nvidia_gpus)
    } else if host.apple_silicon {
        (vec!["Metal".into()], 1)
    } else {
        (vec!["CPU".into()], 0)
    }
}

/// Split `n` layers over `parts` as evenly as possible.
pub fn even_split(n: u64, parts: usize) -> Vec<u64> {
    let parts = parts.max(1) as u64;
    (0..parts)
        .map(|i| n / parts + u64::from(i < n % parts))
        .collect()
}

/// Lay `counts` layers out back to back.
pub fn ranges(counts: &[u64]) -> Vec<(u64, u64)> {
    let mut at = 0;
    counts
        .iter()
        .map(|&c| {
            let r = (at, at + c);
            at += c;
            r
        })
        .collect()
}

pub fn gib(b: u64) -> String {
    Bytes(b).to_string()
}

/// A model name that is safe as a file path component.
pub fn safe_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('.')
        .to_string()
}

/// A shell-friendly rendering of a command line, for logs and docs.
pub fn command_line(cmd: &[String]) -> String {
    cmd.iter()
        .map(|a| {
            if a.chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_./:=,{}@".contains(c))
            {
                a.clone()
            } else {
                format!("'{}'", a.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits() {
        assert_eq!(even_split(10, 3), vec![4, 3, 3]);
        assert_eq!(ranges(&[4, 3, 3]), vec![(0, 4), (4, 7), (7, 10)]);
        assert_eq!(safe_name("Qwen/Qwen2.5:7b"), "Qwen-Qwen2.5-7b");
    }
}
