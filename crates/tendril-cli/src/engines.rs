//! `tendril engines`: the inference engines Tendril can run, whether this
//! machine has each one, and how to install the rest.

use crate::ui::*;
use anyhow::Result;
use clap::Args;
use tendril_engines::{MultiMachine, Security};

#[derive(Args, Debug)]
pub struct EnginesArgs {
    #[arg(long)]
    pub json: bool,
}

pub fn run(a: EnginesArgs) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let found = rt.block_on(tendril_engines::detect_all());
    let engines = tendril_engines::all();
    if a.json {
        let v: Vec<serde_json::Value> = engines
            .iter()
            .zip(&found)
            .map(|(e, d)| serde_json::json!({"info": e.info(), "here": d}))
            .collect();
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    println!();
    println!("{}", bold("Inference engines"));
    for (e, d) in engines.iter().zip(&found) {
        let i = e.info();
        let split = match i.multi_machine {
            MultiMachine::Single => "one machine per model",
            MultiMachine::Layers => "splits layers across any machines",
            MultiMachine::Homogeneous => "splits across identical GPU machines",
            MultiMachine::EngineManaged => "forms its own cluster",
        };
        println!();
        println!(
            "{} {} {}",
            if d.ready { ok_mark() } else { dim("○") },
            bold(&i.name),
            dim(format!("--engine {} · {} · {split}", i.id, i.platforms))
        );
        println!("    {}", i.summary);
        if d.ready {
            let mut what = vec![];
            if let Some(v) = &d.version {
                what.push(format!("version {v}"));
            }
            if !d.accelerators.is_empty() {
                what.push(d.accelerators.join("+"));
            }
            println!("    {} {}", green("installed"), dim(what.join(" · ")));
        }
        for m in &d.missing {
            println!(
                "    {} {} {}",
                if d.ready {
                    yellow("optional")
                } else {
                    yellow("missing ")
                },
                bold(&m.name),
                dim(format!("— {}", m.why))
            );
            if let Some(c) = &m.install {
                println!("             {}", cyan(c));
            }
        }
        for n in &d.notes {
            println!("    {}", dim(n));
        }
        if let Security::TrustedNetworkOnly { .. } = i.security {
            println!(
                "    {}",
                dim("split across machines only on a trusted network (its traffic is unencrypted)")
            );
        }
        println!(
            "    {} {}",
            dim("e.g."),
            dim(format!(
                "tendril load {} --engine {}",
                i.example_model, i.id
            ))
        );
    }
    println!();
    println!(
        "{}",
        dim("Installed one while Tendril runs? The dashboard's “Check again” (or restarting `tendril join`) picks it up.")
    );
    Ok(())
}
