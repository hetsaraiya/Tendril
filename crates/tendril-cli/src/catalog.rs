//! `tendril models` and `tendril hardware`: what Tendril knows offline.

use crate::ui::*;
use anyhow::Result;
use clap::Args;
use tendril_core::model::catalog::CATALOG;
use tendril_core::presets::CHIPS;
use tendril_core::units::{fmt_count, fmt_tokens};
use tendril_core::Quant;

#[derive(Args, Debug)]
pub struct ModelsArgs {
    /// Filter by name or family.
    pub filter: Option<String>,
}

pub fn models(a: ModelsArgs) -> Result<()> {
    let f = a.filter.map(|s| s.to_ascii_lowercase());
    let mut t = Table::new(&[
        "NAME",
        "FAMILY",
        "PARAMS",
        "BF16",
        "Q8_0",
        "Q4_K",
        "CONTEXT",
        "HUGGINGFACE",
    ])
    .right(&[2, 3, 4, 5, 6]);
    for e in CATALOG {
        if let Some(f) = &f {
            if !e.alias.contains(f.as_str())
                && !e.family.to_ascii_lowercase().contains(f.as_str())
                && !e.repo.to_ascii_lowercase().contains(f.as_str())
            {
                continue;
            }
        }
        let s = e.spec();
        t.row(vec![
            bold(e.alias),
            e.family.to_string(),
            fmt_count(s.total_params()),
            format!("{:.1} GiB", s.weight_bytes().as_gib()),
            format!(
                "{:.1} GiB",
                s.with_repr(Quant::Q8_0).weight_bytes().as_gib()
            ),
            format!("{:.1} GiB", s.with_repr(Quant::Q4K).weight_bytes().as_gib()),
            fmt_tokens(s.max_position),
            dim(e.repo),
        ]);
    }
    println!();
    t.print();
    println!();
    println!("{}", dim("Any HuggingFace repo (org/name), local folder or .gguf file works too; these are just available offline."));
    Ok(())
}

pub fn hardware() -> Result<()> {
    let mut t = Table::new(&[
        "KEY",
        "HARDWARE",
        "BACKEND",
        "BANDWIDTH",
        "FP16",
        "DEFAULT MEMORY",
    ])
    .right(&[3, 4, 5]);
    for c in CHIPS {
        t.row(vec![
            bold(c.key),
            c.display.to_string(),
            c.backend.label().to_string(),
            format!("{:.0} GB/s", c.bandwidth_gbs),
            format!("{:.1} TF", c.tflops),
            format!(
                "{:.0} GiB{}",
                c.default_mem_gib,
                if c.fixed_mem { " VRAM" } else { "" }
            ),
        ]);
    }
    println!();
    t.print();
    println!();
    println!(
        "{}",
        dim(
            "Use as --node name=KEY[:memory], e.g. --node studio=m2-ultra:192gb --node gpu=rtx4090"
        )
    );
    Ok(())
}
