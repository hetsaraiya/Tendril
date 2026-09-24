//! `tendril inspect`: what is inside a model, without downloading it.

use crate::common::load_model;
use crate::ui::{self, *};
use anyhow::Result;
use clap::Args;
use tendril_core::model::{Attention, Quant};
use tendril_core::units::{fmt_count, fmt_tokens, Bytes};

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Model: HuggingFace id, local folder or .gguf, or a catalog name.
    pub model: String,
    /// Show per-layer sizes.
    #[arg(long)]
    pub layers: bool,
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub offline: bool,
}

pub fn run(a: InspectArgs) -> Result<()> {
    let m = load_model(&a.model, a.offline, None)?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&m)?);
        return Ok(());
    }
    println!();
    println!("{} {}", bold("Model ·"), bold(cyan(&m.id)));
    let sliding = m
        .attention
        .iter()
        .filter(|x| matches!(x, Attention::Sliding(_)))
        .count();
    let window = m.attention.iter().find_map(|x| match x {
        Attention::Sliding(w) => Some(*w),
        _ => None,
    });
    ui::kv(&[
        (
            "Architecture",
            format!(
                "{} ({}){}",
                m.arch.label(),
                m.model_type,
                if m.arch.executable() {
                    String::new()
                } else {
                    format!(" {}", yellow("— planning only"))
                }
            ),
        ),
        ("Parameters", fmt_count(m.total_params())),
        (
            "Stored as",
            format!(
                "{} · {} of weights{}",
                m.stored.label(),
                m.weight_bytes(),
                if m.bytes_measured {
                    dim(" (measured from tensor headers)")
                } else {
                    dim(" (computed from config)")
                }
            ),
        ),
        (
            "Layers",
            format!(
                "{} × (hidden {}, MLP {})",
                m.num_layers, m.hidden_size, m.intermediate_size
            ),
        ),
        (
            "Attention",
            format!(
                "{} query heads, {} KV heads × {} dims{}",
                m.num_heads,
                m.num_kv_heads,
                m.head_dim,
                match window {
                    Some(w) => format!(
                        " · {sliding}/{} layers use a {}-token sliding window",
                        m.num_layers,
                        fmt_tokens(w)
                    ),
                    None => String::new(),
                }
            ),
        ),
        (
            "Vocabulary",
            format!(
                "{} tokens{}",
                fmt_count(m.vocab_size),
                if m.tie_embeddings {
                    " · output head tied to embeddings"
                } else {
                    ""
                }
            ),
        ),
        (
            "Max context",
            format!("{} tokens", fmt_tokens(m.max_position)),
        ),
        ("Source", m.origin.clone()),
    ]);
    for n in &m.notes {
        println!("  {} {}", warn_mark(), dim(n));
    }

    heading("Where the bytes are");
    let total = m.weight_bytes();
    let layers: Bytes = m.bytes.layers.iter().copied().sum();
    let mut t = Table::new(&["COMPONENT", "SIZE", "SHARE"]).right(&[1, 2]);
    let share = |b: Bytes| format!("{:.1}%", b.0 as f64 / total.0.max(1) as f64 * 100.0);
    t.row(vec![
        "Embeddings".into(),
        m.bytes.embed.to_string(),
        share(m.bytes.embed),
    ]);
    t.row(vec![
        format!("{} transformer layers", m.num_layers),
        layers.to_string(),
        share(layers),
    ]);
    t.row(vec![
        "  per layer (avg)".into(),
        Bytes(layers.0 / m.num_layers.max(1)).to_string(),
        String::new(),
    ]);
    t.row(vec![
        "Final norm".into(),
        m.bytes.final_norm.to_string(),
        share(m.bytes.final_norm),
    ]);
    t.row(vec![
        "Output head".into(),
        if m.tie_embeddings {
            dim("tied (reuses embeddings)")
        } else {
            m.bytes.lm_head.to_string()
        },
        if m.tie_embeddings {
            String::new()
        } else {
            share(m.bytes.lm_head)
        },
    ]);
    t.row(vec![bold("Total"), bold(total.to_string()), String::new()]);
    t.print();

    if a.layers {
        heading("Per-layer");
        let mut t = Table::new(&["LAYER", "SIZE", "ATTENTION"]).right(&[1]);
        for (i, b) in m.bytes.layers.iter().enumerate() {
            let att = match m.attention.get(i) {
                Some(Attention::Sliding(w)) => format!("sliding {}", fmt_tokens(*w)),
                _ => "global".into(),
            };
            t.row(vec![i.to_string(), b.to_string(), att]);
        }
        t.print();
    }

    heading("KV cache (f16) per conversation");
    let all = 0..m.num_layers as usize;
    let mut t = Table::new(&["CONTEXT", "KV CACHE", "WEIGHTS + KV"]).right(&[1, 2]);
    for ctx in [2048u64, 8192, 32768, 131072] {
        if ctx > m.max_position * 2 {
            continue;
        }
        let kv = m.kv_bytes(all.clone(), ctx, 1, 2);
        t.row(vec![
            fmt_tokens(ctx),
            kv.to_string(),
            (kv + total).to_string(),
        ]);
    }
    t.print();

    heading("Representations");
    let mut t = Table::new(&["TYPE", "WEIGHTS", "QUALITY"]).right(&[1]);
    for q in [Quant::Bf16, Quant::Q8_0, Quant::Q6K, Quant::Q4K] {
        if m.stored.is_quantized() && q.bits_per_weight() > m.stored.bits_per_weight() {
            continue;
        }
        let r = m.with_repr(q);
        let label = if q == m.stored || (q == Quant::Bf16 && matches!(m.stored, Quant::F16)) {
            format!("{} {}", q.label(), dim("(stored)"))
        } else {
            q.label().to_string()
        };
        t.row(vec![
            label,
            r.weight_bytes().to_string(),
            dim(q.quality_note()),
        ]);
    }
    t.print();
    println!();
    println!(
        "{}",
        dim(format!(
            "Next: tendril plan {} --node a=m4:16 --node b=m5:16",
            a.model
        ))
    );
    Ok(())
}
