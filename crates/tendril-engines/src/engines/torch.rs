//! vLLM and SGLang: PyTorch servers for HuggingFace models on NVIDIA GPUs.
//!
//! Across machines they split a model by pipeline stages (tensor parallel
//! inside each machine) and talk torch.distributed/NCCL on ports they pick
//! at run time. Tendril cannot carry that traffic, so a model spanning
//! machines needs the user to confirm the machines share a trusted network.

use crate::detect::{find_program, first_version, python, python_package, run_capture};
use crate::engine::*;
use crate::util::*;
use crate::{hf, spec};
use anyhow::{bail, Result};
use async_trait::async_trait;
use std::time::Duration;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Vllm,
    Sglang,
}

pub struct Vllm;
pub struct Sglang;

impl Kind {
    fn id(self) -> &'static str {
        match self {
            Kind::Vllm => "vllm",
            Kind::Sglang => "sglang",
        }
    }

    fn info(self) -> EngineInfo {
        let (name, summary, home, module, risk) = match self {
            Kind::Vllm => (
                "vLLM",
                "High-throughput serving of HuggingFace models on NVIDIA GPUs (continuous batching, paged attention); pipeline-parallel across identical GPU machines.",
                "https://github.com/vllm-project/vllm",
                "vllm",
                "vLLM's docs warn this traffic is unencrypted and that an attacker on the network may be able to run code on the nodes.",
            ),
            Kind::Sglang => (
                "SGLang",
                "Fast serving of HuggingFace models on NVIDIA GPUs (RadixAttention prefix cache); pipeline-parallel across identical GPU machines.",
                "https://github.com/sgl-project/sglang",
                "sglang",
                "The traffic is unencrypted and unauthenticated; anyone on the network path can read or tamper with it.",
            ),
        };
        EngineInfo {
            id: self.id().into(),
            name: name.into(),
            summary: summary.into(),
            homepage: home.into(),
            tasks: vec![Task::Chat],
            inputs: vec![Modality::Text, Modality::Image],
            outputs: vec![Modality::Text],
            multi_machine: MultiMachine::Homogeneous,
            security: Security::TrustedNetworkOnly {
                what: format!("{name} links machines with torch.distributed/NCCL on ports it opens at run time, which Tendril cannot wrap in its encryption."),
                risk: format!("{risk} Only split a model across machines on a private network you trust (same switch/VLAN, no strangers)."),
            },
            platforms: "Linux with NVIDIA GPUs".into(),
            model_hint: "A HuggingFace repo (org/name) with safetensors weights, or a model directory on this server".into(),
            example_model: "Qwen/Qwen2.5-7B-Instruct".into(),
            requirements: vec![Requirement::new(
                module,
                &format!("the {name} Python package runs the model"),
                &[
                    ("linux", &format!("pip install {module}{}", if self == Kind::Sglang { "[all]" } else { "" })),
                    ("macos", "not available on macOS — use llama.cpp, MLX-LM or Ollama"),
                    ("windows", "use WSL2 with an NVIDIA GPU, then: pip install vllm"),
                ],
            )],
        }
    }

    async fn locate(self) -> Option<Located> {
        match self {
            Kind::Vllm => {
                if let Some(p) = find_program(&["vllm"]) {
                    return Some(Located {
                        programs: vec![(
                            "server".into(),
                            vec![p.display().to_string(), "serve".into()],
                        )],
                        ..Default::default()
                    });
                }
                let py = python()?;
                let v = python_package(&py, "vllm").await?;
                Some(Located {
                    programs: vec![(
                        "server".into(),
                        vec![
                            py.display().to_string(),
                            "-m".into(),
                            "vllm.entrypoints.cli.main".into(),
                            "serve".into(),
                        ],
                    )],
                    version: Some(v),
                    ..Default::default()
                })
            }
            Kind::Sglang => {
                let py = python()?;
                let v = python_package(&py, "sglang").await?;
                Some(Located {
                    programs: vec![(
                        "server".into(),
                        vec![
                            py.display().to_string(),
                            "-m".into(),
                            "sglang.launch_server".into(),
                        ],
                    )],
                    version: Some(v),
                    ..Default::default()
                })
            }
        }
    }

    async fn detect(self, host: &HostFacts) -> Detection {
        let info = self.info();
        let Some(loc) = self.locate().await else {
            return not_installed(&info, host);
        };
        let mut version = loc.version.clone();
        if version.is_none() {
            if let Some(cmd) = loc.program("server") {
                let mut c = vec![cmd[0].clone(), "--version".into()];
                if self == Kind::Sglang {
                    c = cmd.clone();
                    c.push("--version".into());
                }
                version = run_capture(&c, Duration::from_secs(60))
                    .await
                    .and_then(|o| first_version(&o));
            }
        }
        let (accelerators, devices) = host_accelerators(host);
        let mut d = Detection {
            engine: self.id().into(),
            ready: true,
            version,
            accelerators,
            devices: host.nvidia_gpus,
            roles: vec!["serve".into()],
            ..Default::default()
        };
        let _ = devices;
        if host.nvidia_gpus > 0 {
            d.roles.push("split-worker".into());
        } else {
            d.notes.push(format!(
                "No NVIDIA GPU here: {} needs one (CPU builds are very slow).",
                info.name
            ));
        }
        d
    }

    async fn resolve(self, req: &ModelRequest) -> Result<ResolvedModel> {
        let m = req.model.trim().to_string();
        let local = std::path::Path::new(&m);
        let (spec, size, id, local_path, vision) = if local.is_dir() {
            let dir = local.to_path_buf();
            let spec = tokio::task::spawn_blocking(move || {
                tendril_core::model::source::inspect_local_dir(&dir)
            })
            .await??;
            let cfg: serde_json::Value = std::fs::read(local.join("config.json"))
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
                .unwrap_or_default();
            let size = spec.weight_bytes().0;
            (spec, size, m.clone(), Some(m.clone()), is_vlm(&cfg))
        } else if hf::is_repo_id(&m) {
            let (repo, rev) = hf::split_repo(&m);
            let files = hf::list_files(&repo, &rev).await?;
            let weights: u64 = files
                .iter()
                .filter(|f| {
                    f.path.ends_with(".safetensors")
                        || f.path.ends_with(".bin") && f.path.contains("pytorch_model")
                })
                .map(|f| f.size)
                .sum();
            if weights == 0 {
                bail!(
                    "{repo} has no safetensors weights{}",
                    if files.iter().any(|f| f.path.ends_with(".gguf")) {
                        " (it is a GGUF repo — use llama.cpp or Ollama)"
                    } else {
                        ""
                    }
                );
            }
            let cfg = hf::json(&repo, &rev, "config.json")
                .await
                .unwrap_or_default();
            let r2 = format!("{repo}@{rev}");
            let spec = tokio::task::spawn_blocking(move || {
                let r = tendril_core::model::source::resolve(&r2)?;
                tendril_core::model::source::inspect(&r, &Default::default())
            })
            .await?
            .unwrap_or_else(|_| {
                spec::opaque(
                    &repo,
                    "huggingface (size only)",
                    weights,
                    spec::working_memory(weights),
                )
            });
            (spec, weights, repo, None, is_vlm(&cfg))
        } else {
            bail!(
                "{} needs a HuggingFace repo (org/name) or a model directory on this server",
                self.info().name
            );
        };
        let layered = spec.num_layers > 1;
        Ok(ResolvedModel {
            engine: self.id().into(),
            served_name: id.clone(),
            id,
            task: Task::Chat,
            vision,
            layered,
            notes: vec![format!("{} of weights; {} downloads them itself from HuggingFace on each machine it runs on", gib(size), self.info().name)],
            spec,
            files: vec![],
            local_path,
            params: serde_json::json!({}),
        })
    }

    fn arrange(
        self,
        _m: &ResolvedModel,
        mut p: Placement,
        machines: &[Machine],
    ) -> Result<Placement> {
        let find = |key: &str| machines.iter().find(|m| m.key == key);
        if p.stages.len() > 1 {
            let devs: Vec<u32> = p
                .stages
                .iter()
                .map(|s| find(&s.machine).map(|m| m.detection.devices).unwrap_or(0))
                .collect();
            if devs.contains(&0) {
                bail!(
                    "{} splits a model only across machines with NVIDIA GPUs",
                    self.info().name
                );
            }
            if devs.iter().any(|d| *d != devs[0]) {
                bail!("{} needs the same number of GPUs on every machine it splits a model across (got {:?})", self.info().name, devs);
            }
            if self == Kind::Sglang {
                // SGLang splits layers evenly between pipeline stages.
                let counts = even_split(p.total_layers, p.stages.len());
                for (s, r) in p.stages.iter_mut().zip(ranges(&counts)) {
                    s.layers = r;
                }
            }
        }
        p.head = 0;
        Ok(p)
    }

    fn roles(self, m: &ResolvedModel, p: &Placement, machines: &[Machine]) -> Result<Vec<Role>> {
        let find = |key: &str| machines.iter().find(|x| x.key == key);
        let tp = find(&p.stages[0].machine)
            .map(|x| x.detection.devices.max(1))
            .unwrap_or(1);
        let pp = p.stages.len();
        let model = m.local_path.clone().unwrap_or_else(|| m.id.clone());
        let ctx = p.context.max(512).to_string();
        let conc = p.concurrency.max(1).to_string();
        let mut roles = Vec::new();
        for (rank, s) in p.stages.iter().enumerate() {
            // Rank 0 hosts the rendezvous on its own LAN address and port;
            // the other ranks reach it once it has started.
            let (master_host, master_port) = if rank == 0 {
                ("{lan_ip}", "{port:master}")
            } else {
                ("{peer_host:server}", "{peer_port:server:master}")
            };
            let mut l = launch(self.id(), "server", &[]);
            match self {
                Kind::Vllm => {
                    l = l
                        .arg(model.clone())
                        .args(["--tensor-parallel-size".to_string(), tp.to_string()])
                        .args(["--max-model-len", &ctx, "--max-num-seqs", &conc]);
                    if rank == 0 {
                        l = l.args([
                            "--host",
                            "127.0.0.1",
                            "--port",
                            "{port}",
                            "--served-model-name",
                            &m.served_name,
                        ]);
                    }
                    if pp > 1 {
                        l = l
                            .args(["--pipeline-parallel-size".to_string(), pp.to_string()])
                            .args([
                                "--nnodes".to_string(),
                                pp.to_string(),
                                "--node-rank".to_string(),
                                rank.to_string(),
                            ])
                            .args([
                                "--master-addr".to_string(),
                                master_host.to_string(),
                                "--master-port".to_string(),
                                master_port.to_string(),
                            ])
                            .env("VLLM_HOST_IP", "{lan_ip}")
                            .env(
                                "VLLM_PP_LAYER_PARTITION",
                                p.stages
                                    .iter()
                                    .map(|s| (s.layers.1 - s.layers.0).to_string())
                                    .collect::<Vec<_>>()
                                    .join(","),
                            );
                        if rank > 0 {
                            l = l.arg("--headless");
                        }
                    }
                }
                Kind::Sglang => {
                    l = l
                        .args(["--model-path".to_string(), model.clone()])
                        .args(["--tp".to_string(), tp.to_string()])
                        .args(["--context-length", &ctx, "--max-running-requests", &conc])
                        .args([
                            "--host",
                            "127.0.0.1",
                            "--port",
                            "{port}",
                            "--served-model-name",
                            &m.served_name,
                        ]);
                    if pp > 1 {
                        l = l
                            .args(["--pp-size".to_string(), pp.to_string()])
                            .args([
                                "--nnodes".to_string(),
                                pp.to_string(),
                                "--node-rank".to_string(),
                                rank.to_string(),
                            ])
                            .args([
                                "--dist-init-addr".to_string(),
                                format!("{master_host}:{master_port}"),
                            ]);
                    }
                }
            }
            if rank == 0 {
                if pp > 1 {
                    l.ports.push("master".into());
                    l.lan = true;
                }
                l.ready_timeout = 7200;
                roles.push(role("server", &s.machine, true, l));
            } else {
                l.lan = true;
                l.ready = Ready::Started;
                l.ready_timeout = 60;
                let mut r = role(&format!("node-{rank}"), &s.machine, false, l);
                r.after_started = vec!["server".into()];
                r.launch.peers = vec!["server".into()];
                roles.push(r);
            }
        }
        // The head waits for its pipeline peers, so it is only ready after
        // they started; nothing else to order.
        Ok(roles)
    }
}

fn is_vlm(cfg: &serde_json::Value) -> bool {
    cfg.get("vision_config").is_some()
        || cfg
            .get("architectures")
            .and_then(|a| a.as_array())
            .is_some_and(|a| {
                a.iter().any(|x| {
                    x.as_str().is_some_and(|s| {
                        s.contains("VL") || s.contains("Vision") || s.contains("Llava")
                    })
                })
            })
}

macro_rules! torch_engine {
    ($t:ty, $k:expr) => {
        #[async_trait]
        impl Engine for $t {
            fn info(&self) -> EngineInfo {
                $k.info()
            }
            async fn locate(&self) -> Option<Located> {
                $k.locate().await
            }
            async fn detect(&self, host: &HostFacts) -> Detection {
                $k.detect(host).await
            }
            async fn resolve(&self, req: &ModelRequest) -> Result<ResolvedModel> {
                $k.resolve(req).await
            }
            fn arrange(
                &self,
                m: &ResolvedModel,
                p: Placement,
                machines: &[Machine],
            ) -> Result<Placement> {
                $k.arrange(m, p, machines)
            }
            fn roles(
                &self,
                m: &ResolvedModel,
                p: &Placement,
                machines: &[Machine],
            ) -> Result<Vec<Role>> {
                $k.roles(m, p, machines)
            }
            fn transfer_note(&self, _m: &ResolvedModel, _p: &Placement, _stage: usize) -> String {
                format!(
                    "{} downloads the model from HuggingFace on this machine",
                    $k.info().name
                )
            }
        }
    };
}

torch_engine!(Vllm, Kind::Vllm);
torch_engine!(Sglang, Kind::Sglang);

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(key: &str, n: u32) -> Machine {
        Machine {
            key: key.into(),
            name: key.into(),
            address: "10.0.0.2".into(),
            local: false,
            detection: Detection {
                engine: "vllm".into(),
                ready: true,
                devices: n,
                roles: vec!["serve".into(), "split-worker".into()],
                ..Default::default()
            },
        }
    }

    #[test]
    fn vllm_pipeline_roles() {
        let ms = vec![gpu("a", 2), gpu("b", 2)];
        let m = ResolvedModel {
            engine: "vllm".into(),
            id: "Qwen/Q".into(),
            served_name: "Qwen/Q".into(),
            task: Task::Chat,
            vision: false,
            spec: spec::opaque("q", "t", 1, 0),
            layered: true,
            files: vec![],
            local_path: None,
            params: serde_json::json!({}),
            notes: vec![],
        };
        let p = Placement {
            stages: vec![
                Stage {
                    machine: "a".into(),
                    layers: (0, 10),
                    weight_bytes: 1,
                },
                Stage {
                    machine: "b".into(),
                    layers: (10, 28),
                    weight_bytes: 1,
                },
            ],
            head: 0,
            total_layers: 28,
            context: 8192,
            concurrency: 4,
        };
        let p = Vllm.arrange(&m, p, &ms).unwrap();
        let r = Vllm.roles(&m, &p, &ms).unwrap();
        assert_eq!(r.len(), 2);
        assert!(r[0].serves_api && r[0].launch.ports == vec!["master"]);
        assert!(r[0]
            .launch
            .env
            .contains(&("VLLM_PP_LAYER_PARTITION".into(), "10,18".into())));
        assert!(r[1].launch.args.contains(&"--headless".to_string()));
        assert_eq!(r[1].after_started, vec!["server"]);
        // Rank 0 doesn't refer to itself as a peer (it starts first).
        let head = r[0].launch.args.join(" ");
        assert!(
            head.contains("--master-addr {lan_ip} --master-port {port:master}"),
            "{head}"
        );
        let worker = r[1].launch.args.join(" ");
        assert!(
            worker.contains("--master-addr {peer_host:server}"),
            "{worker}"
        );
        let p2 = Placement {
            stages: vec![
                Stage {
                    machine: "a".into(),
                    layers: (0, 14),
                    weight_bytes: 1,
                },
                Stage {
                    machine: "c".into(),
                    layers: (14, 28),
                    weight_bytes: 1,
                },
            ],
            ..p.clone()
        };
        let ms2 = vec![gpu("a", 2), gpu("c", 1)];
        assert!(Vllm.arrange(&m, p2, &ms2).is_err());
        let p3 = Sglang.arrange(&m, p, &ms).unwrap();
        assert_eq!(p3.stages[1].layers, (14, 28));
    }
}
