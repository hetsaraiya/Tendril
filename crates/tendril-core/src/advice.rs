//! Actionable suggestions: what to change when a model does not fit, and
//! what the current plan leaves on the table.

use crate::cluster::{Cluster, Link};
use crate::hardware::Backend;
use crate::model::{ModelSpec, Quant};
use crate::planner::{self, Plan, PlanOptions, PlanResult, Workload};
use crate::units::{fmt_ms, fmt_tokens, Bytes, GIB, MIB};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdviceKind {
    /// The model does not fit as requested; this change makes it fit.
    MakeItFit,
    /// A faster or safer alternative the user may prefer.
    Improve,
    /// Neutral information worth knowing.
    Info,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Advice {
    pub kind: AdviceKind,
    pub title: String,
    pub detail: String,
    /// Ready-to-run command, when there is one.
    pub command: Option<String>,
}

/// Largest context (tokens) that fits somewhere on the cluster.
pub fn max_context(
    model: &ModelSpec,
    cluster: &Cluster,
    base: &Workload,
    opts: &PlanOptions,
) -> Option<u64> {
    let fits = |ctx: u64| {
        let mut w = base.clone();
        w.context = ctx;
        w.prompt_tokens = w.prompt_tokens.min(ctx / 2).max(1);
        w.output_tokens = w.output_tokens.min(ctx - w.prompt_tokens).max(1);
        planner::feasible(model, cluster, &w, opts)
    };
    let (mut lo, mut hi) = (256u64, model.max_position.max(256));
    if !fits(lo) {
        return None;
    }
    if fits(hi) {
        return Some(hi);
    }
    while hi - lo > 256 {
        let mid = (lo + hi) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // Round down to a friendly multiple.
    let step = if lo >= 8192 { 1024 } else { 256 };
    Some(lo / step * step)
}

pub fn advise(
    model: &ModelSpec,
    cluster: &Cluster,
    workload: &Workload,
    opts: &PlanOptions,
    result: &PlanResult,
    command_prefix: &str,
) -> Vec<Advice> {
    let mut out = Vec::new();
    match &result.selected {
        None => infeasible_advice(model, cluster, workload, opts, command_prefix, &mut out),
        Some(sel) => feasible_advice(
            model,
            cluster,
            workload,
            opts,
            sel,
            command_prefix,
            &mut out,
        ),
    }
    out
}

fn infeasible_advice(
    model: &ModelSpec,
    cluster: &Cluster,
    workload: &Workload,
    opts: &PlanOptions,
    cmd: &str,
    out: &mut Vec<Advice>,
) {
    // 1. Shorter context / fewer concurrent sequences.
    if let Some(ctx) = max_context(model, cluster, workload, opts) {
        out.push(Advice {
            kind: AdviceKind::MakeItFit,
            title: format!("Use a context of {} tokens or less", fmt_tokens(ctx)),
            detail: format!(
                "At {} tokens × {} sequence(s) the KV cache alone is {}. Weights fit; the cache does not.",
                fmt_tokens(workload.context),
                workload.concurrency,
                model.kv_bytes(0..model.num_layers as usize, workload.context, workload.concurrency, workload.kv_elem_bytes)
            ),
            command: Some(format!("{cmd} --context {}", fmt_tokens(ctx))),
        });
    } else if workload.concurrency > 1 {
        let mut w = workload.clone();
        w.concurrency = 1;
        if planner::feasible(model, cluster, &w, opts) {
            out.push(Advice {
                kind: AdviceKind::MakeItFit,
                title: "Serve one request at a time".into(),
                detail: format!(
                    "{} concurrent sequences need {}× the KV cache.",
                    workload.concurrency, workload.concurrency
                ),
                command: Some(format!("{cmd} --concurrency 1")),
            });
        }
    }

    // 2. A smaller representation (never applied silently).
    if !model.repr.is_quantized() || model.repr == Quant::Q8_0 {
        for q in [Quant::Q8_0, Quant::Q6K, Quant::Q4K] {
            if q.bits_per_weight() >= model.repr.bits_per_weight() {
                continue;
            }
            let mq = model.with_repr(q);
            let r = planner::plan(&mq, cluster, workload, opts);
            if let Some(p) = r.selected {
                out.push(Advice {
                    kind: AdviceKind::MakeItFit,
                    title: format!("Quantize to {} ({})", q.label(), fmt_size_change(model.weight_bytes(), mq.weight_bytes())),
                    detail: format!(
                        "Fits on {} at ~{:.1} tok/s. Quality: {}. Tendril only does this when you ask.",
                        p.label(),
                        p.tokens_per_sec,
                        q.quality_note()
                    ),
                    command: Some(format!("{cmd} --quantize {}", q.label())),
                });
                break;
            }
        }
    }

    // 3. macOS GPU wired limit.
    let mac_nodes: Vec<_> = cluster
        .nodes
        .iter()
        .filter(|n| n.backend == Backend::Metal && n.wired_limit.is_none())
        .collect();
    if !mac_nodes.is_empty() {
        let mut raised = cluster.clone();
        for n in raised.nodes.iter_mut() {
            if n.backend == Backend::Metal && n.wired_limit.is_none() {
                // Leave 3.5 GiB (≤32 GiB machines) or 6 GiB for macOS itself.
                let keep = if n.total_memory.0 <= 32 * GIB {
                    Bytes(3584 * MIB)
                } else {
                    Bytes(6 * GIB)
                };
                n.wired_limit = Some(n.total_memory.saturating_sub(keep));
                n.available_memory = None;
                n.recompute_usable();
            }
        }
        if planner::feasible(model, &raised, workload, opts) {
            let cmds: Vec<String> = raised
                .nodes
                .iter()
                .filter(|n| n.backend == Backend::Metal)
                .map(|n| {
                    format!(
                        "sudo sysctl iogpu.wired_limit_mb={}",
                        n.wired_limit.unwrap().0 / MIB
                    )
                })
                .collect();
            let mut cmds = cmds;
            cmds.dedup();
            let first = cmds.first().cloned();
            out.push(Advice {
                kind: AdviceKind::MakeItFit,
                title: "Let macOS give the GPU more memory".into(),
                detail: format!(
                    "By default macOS caps GPU memory at ~2/3 of RAM. Raising the cap on {} makes this fit (resets on reboot; close other apps first).{}",
                    mac_nodes.iter().map(|n| n.name.as_str()).collect::<Vec<_>>().join(", "),
                    if cmds.len() > 1 { format!(" Run on each Mac respectively: {}", cmds.join(" / ")) } else if mac_nodes.len() > 1 { " Run on each Mac:".to_string() } else { String::new() }
                ),
                command: first,
            });
        }
    }

    // 4. Add hardware.
    let r = planner::plan(model, cluster, workload, opts);
    let short = r
        .rejected
        .iter()
        .map(|x| x.shortfall)
        .min()
        .unwrap_or(Bytes::ZERO);
    let need = model.weight_bytes()
        + model.kv_bytes(
            0..model.num_layers as usize,
            workload.context,
            workload.concurrency,
            workload.kv_elem_bytes,
        );
    let have = cluster.total_usable();
    let gap = need.saturating_sub(have).max(short);
    if gap.0 > 0 {
        let suggestion = if gap.as_gib() <= 10.0 {
            "a 16 GB Mac"
        } else if gap.as_gib() <= 20.0 {
            "a 24–32 GB Mac or a 24 GB GPU"
        } else if gap.as_gib() <= 45.0 {
            "a 64 GB Mac"
        } else {
            "a large-memory machine (128 GB+)"
        };
        out.push(Advice {
            kind: AdviceKind::MakeItFit,
            title: format!("Add at least {} of usable memory", gap.scale(1.1)),
            detail: format!(
                "The cluster has {have} usable; this model at this context needs roughly {} more. For example, add {suggestion} and plan again.",
                gap
            ),
            command: Some(format!("{cmd} --node new=m4:16")),
        });
    }
}

fn fmt_size_change(from: Bytes, to: Bytes) -> String {
    format!("{from} → {to}")
}

fn feasible_advice(
    model: &ModelSpec,
    cluster: &Cluster,
    workload: &Workload,
    opts: &PlanOptions,
    sel: &Plan,
    cmd: &str,
    out: &mut Vec<Advice>,
) {
    // Context headroom on the selected cluster.
    if let Some(ctx) = max_context(model, cluster, workload, opts) {
        if ctx > workload.context {
            out.push(Advice {
                kind: AdviceKind::Info,
                title: format!("Context can go up to ~{} tokens", fmt_tokens(ctx)),
                detail: format!(
                    "You asked for {}; memory allows up to {} at {} concurrent sequence(s).",
                    fmt_tokens(workload.context),
                    fmt_tokens(ctx),
                    workload.concurrency
                ),
                command: None,
            });
        }
    }

    // Network share of latency.
    if sel.stages.len() > 1 {
        let share = sel.network_ms / sel.decode_ms;
        if share > 0.15 {
            let tb = Link::preset("thunderbolt").unwrap();
            let mut faster = cluster.clone();
            faster.links.clear();
            faster.default_link = tb.clone();
            let r = planner::plan(model, &faster, workload, opts);
            if let Some(p) = r.selected {
                if p.decode_ms < sel.decode_ms * 0.9 {
                    out.push(Advice {
                        kind: AdviceKind::Improve,
                        title: format!("The network costs {:.0}% of every token", share * 100.0),
                        detail: format!(
                            "{} per token goes over the link. A Thunderbolt/10 GbE cable would bring decode to ~{} ({:.1} tok/s).",
                            fmt_ms(sel.network_ms),
                            fmt_ms(p.decode_ms),
                            p.tokens_per_sec
                        ),
                        command: Some(format!("{cmd} --link thunderbolt")),
                    });
                }
            }
        }

        // Would a quantized model on one node be faster?
        if !model.repr.is_quantized() {
            for q in [Quant::Q8_0, Quant::Q4K] {
                let mq = model.with_repr(q);
                let r = planner::plan(&mq, cluster, workload, opts);
                if let Some(p) = r.selected.filter(|p| p.stages.len() < sel.stages.len()) {
                    out.push(Advice {
                        kind: AdviceKind::Improve,
                        title: format!("{} fits on fewer machines", q.label()),
                        detail: format!(
                            "{} would run on {} at ~{:.1} tok/s (vs {:.1} now). Quality: {}.",
                            q.label(),
                            p.label(),
                            p.tokens_per_sec,
                            sel.tokens_per_sec,
                            q.quality_note()
                        ),
                        command: Some(format!("{cmd} --quantize {}", q.label())),
                    });
                    break;
                }
            }
        }
    }

    if sel.max_pressure > 0.9 {
        out.push(Advice {
            kind: AdviceKind::Info,
            title: "Memory is tight".into(),
            detail: format!(
                "The fullest node is at {:.0}% of its budget. Close other apps before serving, or plan with --goal memory for more headroom.",
                sel.max_pressure * 100.0
            ),
            command: Some(format!("{cmd} --goal memory")),
        });
    }

    if cluster.nodes.iter().any(|n| n.on_battery == Some(true)) {
        out.push(Advice {
            kind: AdviceKind::Info,
            title: "A machine is on battery".into(),
            detail: "Laptops throttle on battery; expect lower and less stable speed. Plug in for serving.".into(),
            command: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::catalog;
    use crate::presets::node_from_spec;

    #[test]
    fn advice_for_too_big() {
        let m = catalog::lookup("gemma-2-9b").unwrap().spec();
        let c = Cluster::new(
            vec![
                node_from_spec("a", "m4:16").unwrap(),
                node_from_spec("b", "m5:16").unwrap(),
            ],
            Link::preset("thunderbolt").unwrap(),
        );
        let w = Workload::new(4096, 1);
        let o = PlanOptions::default();
        let r = planner::plan(&m, &c, &w, &o);
        assert!(r.selected.is_none());
        let adv = advise(&m, &c, &w, &o, &r, "tendril plan gemma-2-9b");
        let titles: Vec<&str> = adv.iter().map(|a| a.title.as_str()).collect();
        assert!(
            titles.iter().any(|t| t.contains("Quantize to q8_0")),
            "{titles:?}"
        );
        assert!(
            titles.iter().any(|t| t.contains("GPU more memory")),
            "{titles:?}"
        );
        assert!(
            titles.iter().any(|t| t.contains("Add at least")),
            "{titles:?}"
        );
    }

    #[test]
    fn max_context_monotone() {
        let m = catalog::lookup("llama-3.1-8b").unwrap().spec();
        let c = Cluster::new(
            vec![node_from_spec("g", "rtx4090").unwrap()],
            Link::preset("gbe").unwrap(),
        );
        let o = PlanOptions::default();
        let ctx = max_context(&m, &c, &Workload::new(4096, 1), &o).unwrap();
        assert!(ctx > 8192 && ctx < 131072, "{ctx}");
        let ctx4 = max_context(&m, &c, &Workload::new(4096, 4), &o).unwrap();
        assert!(ctx4 < ctx);
    }
}
