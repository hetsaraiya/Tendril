//! `tendril fit`: a quick "can I run it?" matrix of representation × context.

use crate::common::{load_model, ClusterArgs};
use crate::ui::*;
use anyhow::Result;
use clap::Args;
use tendril_core::model::Quant;
use tendril_core::units::fmt_tokens;
use tendril_core::{plan, Goal, PlanOptions, Workload};

#[derive(Args, Debug)]
pub struct FitArgs {
    pub model: String,
    #[command(flatten)]
    pub cluster: ClusterArgs,
    /// Conversations served at the same time.
    #[arg(long, default_value_t = 1)]
    pub concurrency: u64,
    #[arg(long)]
    pub offline: bool,
    #[arg(long)]
    pub json: bool,
}

pub fn run(a: FitArgs) -> Result<()> {
    let base = load_model(&a.model, a.offline, None)?;
    let cluster = a.cluster.build()?;
    let opts = PlanOptions {
        goal: Goal::Balanced,
        ..Default::default()
    };
    let mut contexts: Vec<u64> = [2048u64, 8192, 32768, 131072]
        .into_iter()
        .filter(|&c| c <= base.max_position)
        .collect();
    if contexts.last() != Some(&base.max_position) && base.max_position < 131072 {
        contexts.push(base.max_position);
    }
    let mut reprs = vec![base.stored];
    for q in [Quant::Q8_0, Quant::Q6K, Quant::Q4K] {
        if q.bits_per_weight() < base.stored.bits_per_weight() {
            reprs.push(q);
        }
    }
    let mut json_rows = Vec::new();
    let mut headers = vec!["WEIGHTS".to_string(), "SIZE".to_string()];
    headers.extend(contexts.iter().map(|c| format!("{} CTX", fmt_tokens(*c))));
    let hdr_refs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
    let mut t = Table::new(&hdr_refs).right(&[1]);
    for q in &reprs {
        let m = base.with_repr(*q);
        let mut row = vec![
            if *q == base.stored {
                format!("{} {}", q.label(), dim("stored"))
            } else {
                q.label().to_string()
            },
            format!("{:.1} GiB", m.weight_bytes().as_gib()),
        ];
        for &ctx in &contexts {
            let w = Workload::new(ctx, a.concurrency);
            let r = plan(&m, &cluster, &w, &opts);
            match r.selected {
                Some(p) => {
                    row.push(format!(
                        "{} {:>5.1} tok/s {}",
                        ok_mark(),
                        p.tokens_per_sec,
                        dim(if p.stages.len() == 1 {
                            format!("on {}", p.stages[0].node_name)
                        } else {
                            format!("{} machines", p.stages.len())
                        })
                    ));
                    json_rows.push(serde_json::json!({"repr": q.label(), "context": ctx, "fits": true, "tokens_per_sec": p.tokens_per_sec, "nodes": p.node_names()}));
                }
                None => {
                    row.push(format!("{} {}", bad_mark(), dim("doesn't fit")));
                    json_rows.push(
                        serde_json::json!({"repr": q.label(), "context": ctx, "fits": false}),
                    );
                }
            }
        }
        t.row(row);
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&json_rows)?);
        return Ok(());
    }
    println!();
    println!(
        "{} {} {}",
        bold("Can it run? ·"),
        bold(cyan(&base.id)),
        dim(format!(
            "on {} ({} usable, {} conversation{})",
            cluster
                .nodes
                .iter()
                .map(|n| n.name.as_str())
                .collect::<Vec<_>>()
                .join(" + "),
            cluster.total_usable(),
            a.concurrency,
            if a.concurrency == 1 { "" } else { "s" }
        ))
    );
    println!();
    t.print();
    println!();
    println!(
        "{}",
        dim(format!(
            "Details for any cell: tendril plan {} --context 8k [--quantize q8_0] --explain",
            a.model
        ))
    );
    Ok(())
}
