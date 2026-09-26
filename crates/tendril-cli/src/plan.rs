//! `tendril plan`: decide and explain how a model should run.

use crate::common::{command_prefix, finish_command, ClusterArgs, WorkloadArgs};
use crate::ui::{self, *};
use anyhow::Result;
use clap::Args;
use tendril_core::advice::{self, AdviceKind};
use tendril_core::cluster::Cluster;
use tendril_core::model::ModelSpec;
use tendril_core::planner::{Plan, PlanResult};
use tendril_core::units::{fmt_count, fmt_ms, fmt_tokens, Bytes};
use tendril_core::Workload;

#[derive(Args, Debug)]
pub struct PlanArgs {
    /// Model: HuggingFace id (org/name), local folder or .gguf, or a catalog name (see `tendril models`).
    pub model: String,
    #[command(flatten)]
    pub cluster: ClusterArgs,
    #[command(flatten)]
    pub work: WorkloadArgs,
    /// Show every cut, the memory breakdown and rejected alternatives.
    #[arg(long, short = 'e')]
    pub explain: bool,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

pub fn run(a: PlanArgs) -> Result<()> {
    let model = a.work.model(&a.model)?;
    let cluster = a.cluster.build()?;
    let workload = a.work.workload(&model)?;
    let opts = a.work.options()?;
    let t0 = std::time::Instant::now();
    let result = tendril_core::plan(&model, &cluster, &workload, &opts);
    let prefix = command_prefix("plan", &a.model, &a.cluster, &a.work);
    let adv = advice::advise(&model, &cluster, &workload, &opts, &result, &prefix);
    let elapsed = t0.elapsed();

    if a.json {
        let out = serde_json::json!({
            "model": model,
            "cluster": cluster,
            "result": result,
            "advice": adv,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    header(&model, &cluster, &workload, &a, opts.goal.label());
    match &result.selected {
        Some(p) => render_selected(p, &model, &cluster, &result),
        None => render_infeasible(&model, &cluster, &result),
    }
    if a.explain {
        render_explain(&model, &cluster, &result);
    }
    if !adv.is_empty() {
        heading(if result.selected.is_some() {
            "Suggestions"
        } else {
            "How to make it fit"
        });
        let width = term_width();
        for s in &adv {
            let mark = match s.kind {
                AdviceKind::MakeItFit => cyan("→"),
                AdviceKind::Improve => magenta("↑"),
                AdviceKind::Info => dim("·"),
            };
            println!("  {} {}", mark, bold(&s.title));
            println!("    {}", dim(ui::wrap(&s.detail, 4, width)));
            if let Some(c) = &s.command {
                println!(
                    "    {}",
                    cyan(format!("$ {}", finish_command(c, &a.cluster, &a.work)))
                );
            }
        }
    }
    println!();
    println!(
        "{}",
        dim(format!(
            "Evaluated {} machine orderings and {} layer cuts in {}. Estimates, not measurements{}.",
            result.orders_evaluated,
            fmt_count(result.cuts_evaluated),
            fmt_ms(elapsed.as_secs_f64() * 1000.0),
            if a.explain { "" } else { " — add --explain for the full reasoning" }
        ))
    );
    Ok(())
}

fn header(model: &ModelSpec, cluster: &Cluster, w: &Workload, a: &PlanArgs, goal: &str) {
    println!();
    println!("{} {}", bold("Tendril plan ·"), bold(cyan(&model.id)));
    let arch = format!(
        "{} · {} params · {} · {} weights · {} layers{}",
        model.arch.label(),
        fmt_count(model.total_params()),
        model.repr.label(),
        model.weight_bytes(),
        model.num_layers,
        if model.repr != model.stored {
            format!(
                " {}",
                yellow(format!(
                    "(quantized from {} on request)",
                    model.stored.label()
                ))
            )
        } else {
            String::new()
        }
    );
    let work = format!(
        "{} context · {} conversation{} · ~{} token prompts · goal: {}",
        fmt_tokens(w.context),
        w.concurrency,
        if w.concurrency == 1 { "" } else { "s" },
        fmt_tokens(w.prompt_tokens),
        goal
    );
    let link = if cluster.nodes.len() > 1 {
        let l = &cluster.default_link;
        format!(
            " · link: {} ({:.1} Gb/s, {} RTT){}",
            l.name,
            l.bandwidth_gbps,
            fmt_ms(l.rtt_ms),
            if a.cluster.link.is_none() && a.cluster.cluster.is_none() {
                dim(" — assumed, set --link")
            } else {
                String::new()
            }
        )
    } else {
        String::new()
    };
    let nodes = format!(
        "{} machine{} · {} usable{}",
        cluster.nodes.len(),
        if cluster.nodes.len() == 1 { "" } else { "s" },
        cluster.total_usable(),
        link
    );
    ui::kv(&[("Model", arch), ("Workload", work), ("Cluster", nodes)]);
    for n in &model.notes {
        println!("  {} {}", warn_mark(), dim(n));
    }
}

fn stage_table(p: &Plan, cluster: &Cluster) {
    let mut t = Table::new(&["STAGE", "MACHINE", "RUNS", "MEMORY", "", "DECODE"]).right(&[4, 5]);
    for (i, s) in p.stages.iter().enumerate() {
        let n = &cluster.nodes[s.node];
        let used = s.mem.peak;
        t.row(vec![
            format!("{}", i + 1),
            format!(
                "{} {}",
                bold(&s.node_name),
                dim(format!("({})", short_chip(&n.chip)))
            ),
            s.describe_components(),
            ui::bar(used.0 as f64, s.mem.usable.0 as f64, 16),
            format!("{:.1}/{:.1} GiB", used.as_gib(), s.mem.usable.as_gib()),
            fmt_ms(s.decode_ms),
        ]);
    }
    if p.stages.len() > 1 {
        t.row(vec![
            String::new(),
            dim("network"),
            dim(format!(
                "{} per hop per token + sampled token back",
                Bytes(p.boundary_bytes_per_token)
            )),
            String::new(),
            String::new(),
            dim(fmt_ms(p.network_ms)),
        ]);
    }
    t.print();
}

fn short_chip(c: &str) -> String {
    c.replace("NVIDIA GeForce ", "")
        .replace("NVIDIA ", "")
        .replace("Apple ", "")
}

fn render_selected(p: &Plan, model: &ModelSpec, cluster: &Cluster, r: &PlanResult) {
    println!();
    let headline = if p.stages.len() == 1 {
        format!("Runs on {} alone", p.stages[0].node_name)
    } else {
        format!(
            "Runs across {} machines ({})",
            p.stages.len(),
            p.node_names().join(" → ")
        )
    };
    println!("{} {}", ok_mark(), bold(green(headline)));
    let mut line = format!(
        "  ~{} per conversation · first token in ~{}",
        bold(format!("{:.1} tok/s", p.tokens_per_sec)),
        fmt_ms(p.ttft_ms)
    );
    if r.workload.concurrency > 1 {
        line.push_str(&format!(
            " · ~{:.0} tok/s total at {} concurrent",
            p.throughput_tps, r.workload.concurrency
        ));
    }
    println!("{line}");
    println!();
    stage_table(p, cluster);

    heading("Why this plan");
    let width = term_width();
    let mut bullets: Vec<String> = Vec::new();
    // Single-node rejections explain the need for distribution.
    let singles: Vec<_> = r.rejected.iter().filter(|x| x.nodes.len() == 1).collect();
    if p.stages.len() > 1 && !singles.is_empty() {
        for s in singles.iter().take(3) {
            bullets.push(format!(
                "{} can't hold it alone: {}",
                bold(&s.nodes[0]),
                s.reason
            ));
        }
    }
    if p.stages.len() > 1 {
        let desc: Vec<String> = p
            .stages
            .iter()
            .map(|s| {
                let n = &cluster.nodes[s.node];
                format!(
                    "{} takes {} of {} layers ({:.0} GB/s, {} budget)",
                    s.node_name,
                    s.layers(),
                    model.num_layers,
                    n.bandwidth_gbs,
                    s.mem.usable
                )
            })
            .collect();
        bullets.push(format!(
            "{}. Splits are sized to each machine's speed and memory, not 50/50.",
            desc.join("; ")
        ));
        if p.stages.len() == 2 && !r.cut_table.is_empty() {
            let fastest = r
                .cut_table
                .iter()
                .filter(|c| c.decode_ms.is_some())
                .min_by(|a, b| a.decode_ms.unwrap().total_cmp(&b.decode_ms.unwrap()));
            let feasible = r.cut_table.iter().filter(|c| c.decode_ms.is_some()).count();
            let chosen = r.cut_table.iter().find(|c| c.selected);
            if let (Some(f), Some(c)) = (fastest, chosen) {
                if f.cut != c.cut {
                    let tight = if f.first.pressure() > f.second.pressure() {
                        (&r.cut_table_nodes[0], f.first.pressure())
                    } else {
                        (&r.cut_table_nodes[1], f.second.pressure())
                    };
                    bullets.push(format!(
                        "{feasible} of {} cuts fit. The fastest (after layer {}) is {} quicker but would fill {} to {:.0}% of its budget; this plan keeps more headroom. Use --goal latency to take it.",
                        r.cut_table.len(),
                        f.cut - 1,
                        fmt_ms(c.decode_ms.unwrap_or(0.0) - f.decode_ms.unwrap_or(0.0)),
                        tight.0,
                        tight.1 * 100.0
                    ));
                } else {
                    bullets.push(format!("{feasible} of {} possible cuts fit in memory; this one is the fastest of them.", r.cut_table.len()));
                }
            }
        }
        if model.tie_embeddings {
            bullets.push(format!(
                "Embeddings are tied to the output head, so both ends hold a copy ({}).",
                model.bytes.embed
            ));
        }
        bullets.push(
            "Weights stay put; only the boundary activation (and the sampled token back) crosses the network each token."
                .to_string(),
        );
    }
    for e in &r.excluded {
        bullets.push(format!("{} left out: {}", bold(&e.node), e.reason));
    }
    if let Some(alt) = r.alternatives.first() {
        let d = alt.decode_ms - p.decode_ms;
        bullets.push(format!(
            "Runner-up: {} ({:.1} tok/s, {}{} per token).",
            alt.label(),
            alt.tokens_per_sec,
            if d >= 0.0 { "+" } else { "−" },
            fmt_ms(d.abs())
        ));
    }
    let headroom = p
        .stages
        .iter()
        .map(|s| s.mem.headroom())
        .min()
        .unwrap_or_default();
    bullets.push(format!(
        "Tightest machine keeps {} free ({:.0}% of its budget used at full {} context).",
        headroom,
        p.max_pressure * 100.0,
        fmt_tokens(r.workload.context)
    ));
    for b in bullets {
        println!("  • {}", ui::wrap(&b, 4, width));
    }
}

fn render_infeasible(model: &ModelSpec, cluster: &Cluster, r: &PlanResult) {
    println!();
    println!(
        "{} {}",
        bad_mark(),
        bold(red(format!(
            "{} does not fit on {} at {} context",
            model.id,
            if cluster.nodes.len() == 1 {
                cluster.nodes[0].name.clone()
            } else {
                "these machines".into()
            },
            fmt_tokens(r.workload.context)
        )))
    );
    let width = term_width();
    for rej in r.rejected.iter().take(5) {
        let who = if rej.nodes.len() == 1 {
            format!("{} alone", rej.nodes[0])
        } else {
            rej.nodes.join(" → ")
        };
        println!(
            "  {} {}: {}",
            bad_mark(),
            bold(who),
            ui::wrap(&rej.reason, 6, width)
        );
    }
    if r.rejected.len() > 5 {
        println!(
            "  {}",
            dim(format!(
                "… and {} more orderings (see --explain)",
                r.rejected.len() - 5
            ))
        );
    }
}

fn render_explain(model: &ModelSpec, cluster: &Cluster, r: &PlanResult) {
    if let Some(p) = &r.selected {
        heading("Memory per machine");
        let mut t = Table::new(&[
            "MACHINE", "WEIGHTS", "KV CACHE", "BUFFERS", "RUNTIME", "PEAK", "MARGIN", "BUDGET",
            "HEADROOM",
        ])
        .right(&[1, 2, 3, 4, 5, 6, 7, 8]);
        for s in &p.stages {
            let m = &s.mem;
            t.row(vec![
                s.node_name.clone(),
                m.weights.to_string(),
                m.kv.to_string(),
                (m.scratch + m.transport).to_string(),
                m.runtime.to_string(),
                bold(m.peak.to_string()),
                m.margin.to_string(),
                m.usable.to_string(),
                green(m.headroom().to_string()),
            ]);
        }
        t.print();
        for s in &p.stages {
            let n = &cluster.nodes[s.node];
            println!(
                "  {} {}: {}",
                dim("budget"),
                s.node_name,
                dim(&n.usable_reason)
            );
        }

        heading("Time per token");
        let mut t = Table::new(&["STAGE", "DECODE", "PREFILL (prompt)", "BANDWIDTH USED"])
            .right(&[1, 2, 3]);
        for s in &p.stages {
            let n = &cluster.nodes[s.node];
            t.row(vec![
                s.node_name.clone(),
                fmt_ms(s.decode_ms),
                fmt_ms(s.prefill_ms),
                format!("{:.0} GB/s effective", n.effective_bandwidth_gbs()),
            ]);
        }
        if p.stages.len() > 1 {
            t.row(vec![
                "network".into(),
                fmt_ms(p.network_ms),
                String::new(),
                String::new(),
            ]);
        }
        t.row(vec![
            bold("total"),
            bold(fmt_ms(p.decode_ms)),
            bold(fmt_ms(p.ttft_ms)),
            String::new(),
        ]);
        t.print();

        heading("Score");
        for (k, v) in &p.score_parts {
            println!("  {:<24} {:.3}", dim(k), v);
        }
        println!(
            "  {:<24} {}",
            bold("total (lower is better)"),
            bold(format!("{:.3}", p.score))
        );
    }

    if !r.cut_table.is_empty() {
        heading(&format!(
            "Every cut between {} and {}",
            r.cut_table_nodes.first().cloned().unwrap_or_default(),
            r.cut_table_nodes.get(1).cloned().unwrap_or_default()
        ));
        let a = &r.cut_table_nodes[0];
        let b = &r.cut_table_nodes[1];
        let mut t = Table::new(&[
            "CUT",
            &format!("{a} LAYERS"),
            &format!("{a} PEAK"),
            &format!("{b} PEAK"),
            "DECODE",
            "",
        ])
        .right(&[1, 2, 3, 4]);
        let rows = &r.cut_table;
        // Long tables: show rows around the feasible window and the selection.
        let interesting: Vec<usize> = (0..rows.len())
            .filter(|&i| {
                rows.len() <= 40
                    || rows[i].selected
                    || rows[i].decode_ms.is_some()
                    || i % (rows.len() / 16).max(1) == 0
            })
            .collect();
        for &i in &interesting {
            let row = &rows[i];
            let fa = if row.first.fits() {
                row.first.peak.to_string()
            } else {
                red(format!("{} ✗", row.first.peak))
            };
            let fb = if row.second.fits() {
                row.second.peak.to_string()
            } else {
                red(format!("{} ✗", row.second.peak))
            };
            let status = if row.selected {
                green("◀ selected")
            } else if row.decode_ms.is_none() {
                dim(if !row.first.fits() {
                    format!("{a} over budget")
                } else {
                    format!("{b} over budget")
                })
            } else {
                String::new()
            };
            t.row(vec![
                format!("{}", row.cut),
                format!("0–{}", row.cut - 1),
                fa,
                fb,
                row.decode_ms.map(fmt_ms).unwrap_or_else(|| dim("—")),
                status,
            ]);
        }
        t.print();
        println!(
            "  {}",
            dim(format!(
                "Budgets: {a} {} · {b} {} (after safety margin).",
                cluster
                    .nodes
                    .iter()
                    .find(|n| &n.name == a)
                    .map(|n| n.usable_memory)
                    .unwrap_or_default(),
                cluster
                    .nodes
                    .iter()
                    .find(|n| &n.name == b)
                    .map(|n| n.usable_memory)
                    .unwrap_or_default()
            ))
        );
    }

    if !r.alternatives.is_empty() {
        heading("Other feasible plans");
        let mut t =
            Table::new(&["PLAN", "TOK/S", "FIRST TOKEN", "PEAK USE", "SCORE"]).right(&[1, 2, 3, 4]);
        for p in &r.alternatives {
            t.row(vec![
                p.label(),
                format!("{:.1}", p.tokens_per_sec),
                fmt_ms(p.ttft_ms),
                format!("{:.0}%", p.max_pressure * 100.0),
                format!("{:.3}", p.score),
            ]);
        }
        t.print();
    }

    if !r.rejected.is_empty() {
        heading("Rejected");
        let width = term_width();
        for rej in r.rejected.iter().take(12) {
            let who = if rej.nodes.len() == 1 {
                format!("{} alone", rej.nodes[0])
            } else {
                rej.nodes.join(" → ")
            };
            println!(
                "  {} {}: {}",
                bad_mark(),
                who,
                dim(ui::wrap(&rej.reason, 6, width))
            );
        }
    }

    heading("Assumptions");
    let lines = [
        format!(
            "KV cache in {} bytes/element, {} per token across all layers.",
            r.workload.kv_elem_bytes,
            model.kv_bytes(0..model.num_layers as usize, 1, 1, r.workload.kv_elem_bytes)
        ),
        "Decode is memory-bandwidth bound: every token streams the stage's weights once.".into(),
        format!(
            "Prefill is processed in {}-token chunks that pipeline through the stages.",
            r.workload.prefill_chunk
        ),
        if model.bytes_measured {
            "Weight sizes come from the actual tensor headers.".into()
        } else {
            "Weight sizes are computed from config.json (no tensor headers were read).".into()
        },
        "Speeds are predictions from hardware specs; `tendril node --probe` measures this machine."
            .into(),
    ];
    for l in lines {
        println!("  · {}", dim(l));
    }
}
