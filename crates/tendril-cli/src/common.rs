//! Argument groups shared by several commands.

use anyhow::{bail, Context, Result};
use clap::Args;
use tendril_core::cluster::{load_cluster_file, Cluster, Link};
use tendril_core::hardware::detect_local;
use tendril_core::model::source::{inspect, resolve, InspectOptions};
use tendril_core::model::{ModelSpec, Quant};
use tendril_core::presets::node_from_spec;
use tendril_core::units::parse_tokens;
use tendril_core::{Goal, PlanOptions, Workload};

#[derive(Args, Debug, Clone, Default)]
pub struct ClusterArgs {
    /// Add a machine: `name=hardware[:memory]`, e.g. `--node air=m4:16gb --node box=rtx4090`.
    /// Run `tendril hardware` for the list. Without --node/--cluster, this machine is used.
    #[arg(long = "node", short = 'n', value_name = "NAME=HW")]
    pub nodes: Vec<String>,

    /// Describe machines and links in a TOML file (see examples/).
    #[arg(long, value_name = "FILE")]
    pub cluster: Option<std::path::PathBuf>,

    /// Also include this machine (detected) alongside --node/--cluster.
    #[arg(long)]
    pub with_local: bool,

    /// Network between machines: tb5, thunderbolt, 10gbe, 2.5gbe, gbe, wifi…
    #[arg(long, value_name = "KIND")]
    pub link: Option<String>,

    /// Plan against total memory instead of what is free right now.
    #[arg(long)]
    pub ignore_free: bool,
}

#[derive(Args, Debug, Clone)]
pub struct WorkloadArgs {
    /// Maximum tokens per conversation (prompt + reply), e.g. 8k, 32768.
    #[arg(long, short = 'c', default_value = "8k", value_name = "TOKENS")]
    pub context: String,

    /// Conversations served at the same time.
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub concurrency: u64,

    /// Typical prompt length for time-to-first-token estimates.
    #[arg(long, value_name = "TOKENS")]
    pub prompt: Option<String>,

    /// What to optimize: balanced, latency, throughput, memory.
    #[arg(long, short = 'g', default_value = "balanced")]
    pub goal: String,

    /// Plan with a smaller representation (q8_0, q6_k, q4_k…). Never applied unless asked.
    #[arg(long, short = 'q', value_name = "TYPE")]
    pub quantize: Option<String>,

    /// Keep this fraction of each machine's budget free (default 0.05).
    #[arg(long, default_value_t = 0.05, value_name = "FRACTION")]
    pub safety: f64,

    /// Do not use the network to inspect the model.
    #[arg(long)]
    pub offline: bool,
}

impl ClusterArgs {
    pub fn build(&self) -> Result<Cluster> {
        let default_link = match &self.link {
            Some(k) => Link::preset(k)
                .with_context(|| format!("unknown link '{k}'. Options: {}", Link::names()))?,
            None => Link::preset("gbe").unwrap(),
        };
        let mut cluster = match &self.cluster {
            Some(p) => {
                let mut c = load_cluster_file(p)?;
                if self.link.is_some() {
                    c.default_link = default_link.clone();
                    c.links.clear();
                }
                c
            }
            None => Cluster::new(Vec::new(), default_link),
        };
        for spec in &self.nodes {
            let (name, hw) = match spec.split_once('=') {
                Some((n, h)) => (n.trim().to_string(), h.trim()),
                None => (format!("node{}", cluster.nodes.len() + 1), spec.trim()),
            };
            let n = node_from_spec(&name, hw).map_err(anyhow::Error::msg)?;
            cluster.nodes.push(n);
        }
        if cluster.nodes.is_empty() || self.with_local {
            let local = detect_local(!self.ignore_free);
            cluster.nodes.insert(0, local);
        }
        cluster.dedupe_names();
        if cluster.nodes.len() > 16 {
            bail!("Tendril plans for up to 16 machines at a time");
        }
        Ok(cluster)
    }
}

impl WorkloadArgs {
    pub fn workload(&self, model: &ModelSpec) -> Result<Workload> {
        let ctx = parse_tokens(&self.context)
            .with_context(|| format!("cannot parse context '{}'", self.context))?;
        if ctx < 16 {
            bail!("context must be at least 16 tokens");
        }
        let mut w = Workload::new(ctx, self.concurrency);
        if let Some(p) = &self.prompt {
            let p = parse_tokens(p).with_context(|| format!("cannot parse prompt length '{p}'"))?;
            w.prompt_tokens = p.clamp(1, ctx - 1);
            w.output_tokens = (ctx - w.prompt_tokens).clamp(1, 256);
        }
        if ctx > model.max_position {
            eprintln!(
                "{} {} was trained for {} tokens; planning {} anyway",
                crate::ui::warn_mark(),
                model.id,
                tendril_core::units::fmt_tokens(model.max_position),
                tendril_core::units::fmt_tokens(ctx)
            );
        }
        Ok(w)
    }

    pub fn options(&self) -> Result<PlanOptions> {
        let goal = Goal::parse(&self.goal).with_context(|| {
            format!(
                "unknown goal '{}'. Use balanced, latency, throughput or memory",
                self.goal
            )
        })?;
        if !(0.0..0.5).contains(&self.safety) {
            bail!("--safety must be between 0 and 0.5");
        }
        Ok(PlanOptions {
            goal,
            safety_frac: self.safety,
            ..Default::default()
        })
    }

    pub fn model(&self, name: &str) -> Result<ModelSpec> {
        load_model(name, self.offline, self.quantize.as_deref())
    }
}

pub fn load_model(name: &str, offline: bool, quantize: Option<&str>) -> Result<ModelSpec> {
    let r = resolve(name)?;
    let spec = inspect(&r, &InspectOptions { offline })
        .with_context(|| format!("could not inspect {}", r.describe()))?;
    match quantize {
        None => Ok(spec),
        Some(q) => {
            let q = Quant::parse(q).with_context(|| {
                format!("unknown quantization '{q}' (q8_0, q6_k, q5_k, q4_k, q4_0, f16, bf16)")
            })?;
            if spec.stored.is_quantized() && q.bits_per_weight() > spec.stored.bits_per_weight() {
                bail!(
                    "{} is already stored as {}; re-quantizing to {} would not add precision",
                    spec.id,
                    spec.stored.label(),
                    q.label()
                );
            }
            Ok(spec.with_repr(q))
        }
    }
}

/// Finish a suggested command: re-add flags the suggestion does not override.
pub fn finish_command(cmd: &str, c: &ClusterArgs, w: &WorkloadArgs) -> String {
    let mut s = cmd.to_string();
    if !s.starts_with("tendril ") {
        return s;
    }
    if let Some(l) = &c.link {
        if !s.contains("--link") {
            s.push_str(&format!(" --link {l}"));
        }
    }
    if !s.contains("--context") && w.context != "8k" {
        s.push_str(&format!(" --context {}", w.context));
    }
    if c.nodes.is_empty()
        && c.cluster.is_none()
        && s.contains("--node")
        && !s.contains("--with-local")
    {
        s.push_str(" --with-local");
    }
    if let Some(q) = &w.quantize {
        if !s.contains("--quantize") {
            s.push_str(&format!(" --quantize {q}"));
        }
    }
    s
}

/// Reconstruct a copy-pasteable command prefix for suggestions.
pub fn command_prefix(sub: &str, model: &str, c: &ClusterArgs, w: &WorkloadArgs) -> String {
    let mut s = format!("tendril {sub} {model}");
    for n in &c.nodes {
        s.push_str(&format!(" --node {n}"));
    }
    if let Some(p) = &c.cluster {
        s.push_str(&format!(" --cluster {}", p.display()));
    }
    if c.with_local {
        s.push_str(" --with-local");
    }
    if w.concurrency != 1 {
        s.push_str(&format!(" --concurrency {}", w.concurrency));
    }
    s
}
