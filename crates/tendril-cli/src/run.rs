//! `tendril run`: chat with a model on this machine, in the terminal.

use crate::models::ensure_local;
use crate::ui::*;
use anyhow::{bail, Result};
use clap::Args;
use std::io::{BufRead, Write};
use std::time::Instant;
use tendril_core::units::Bytes;
use tendril_engine::device::{device_from_name, device_label};
use tendril_engine::generate::{GenEvent, GenerateRequest, LocalModel, ModelFiles};
use tendril_engine::linear::WeightFormat;
use tendril_engine::sampler::SamplingParams;
use tendril_engine::tokenizer::ChatMessage;

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Model: HuggingFace id, catalog name or local folder.
    pub model: String,
    /// Answer one prompt and exit (otherwise start an interactive chat).
    #[arg(long, short = 'p')]
    pub prompt: Option<String>,
    /// System prompt.
    #[arg(long)]
    pub system: Option<String>,
    /// Weight format: native, q8_0, q6_k, q4_k.
    #[arg(long, short = 'q', default_value = "native")]
    pub quantize: String,
    /// auto, cpu, metal or cuda.
    #[arg(long, default_value = "auto")]
    pub device: String,
    #[arg(long, default_value_t = 1024)]
    pub max_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,
    #[arg(long)]
    pub seed: Option<u64>,
    /// Don't download; fail if the model is not already local.
    #[arg(long)]
    pub offline: bool,
}

pub fn run(a: RunArgs) -> Result<()> {
    let format = WeightFormat::parse(&a.quantize).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown --quantize '{}' (native, q8_0, q6_k, q4_k)",
            a.quantize
        )
    })?;
    let (dir, name) = ensure_local(&a.model, !a.offline)?;
    let device = device_from_name(&a.device)?;

    // Refuse early, with advice, if it clearly won't fit here.
    let mut spec = tendril_core::model::source::inspect_local_dir(&dir)?;
    spec = match format {
        WeightFormat::Native => spec,
        WeightFormat::Q8_0 => spec.with_repr(tendril_core::Quant::Q8_0),
        WeightFormat::Q4K => spec.with_repr(tendril_core::Quant::Q4K),
        WeightFormat::Q6K => spec.with_repr(tendril_core::Quant::Q6K),
    };
    let mut node = tendril_core::hardware::detect_local(true);
    if matches!(device, candle_core::Device::Cpu) {
        node.backend = tendril_core::Backend::Cpu;
        node.recompute_usable();
    }
    let cluster = tendril_core::Cluster::new(
        vec![node.clone()],
        tendril_core::Link::preset("local").unwrap(),
    );
    let w = tendril_core::Workload::new(4096, 1);
    let plan = tendril_core::plan(&spec, &cluster, &w, &Default::default());
    if plan.selected.is_none() {
        let reason = plan
            .rejected
            .first()
            .map(|r| r.reason.clone())
            .unwrap_or_default();
        bail!(
            "{name} won't fit on this machine: {reason}.\n  Try: tendril run {} --quantize q8_0   (or q4_k)\n  Or spread it over several machines: tendril serve {}",
            a.model,
            a.model
        );
    }

    let t0 = Instant::now();
    eprint!(
        "{} Loading {} ({}) on {}…",
        cyan("◌"),
        bold(&name),
        spec.weight_bytes(),
        device_label(&device)
    );
    std::io::stderr().flush().ok();
    let mut model = LocalModel::load(ModelFiles::new(&dir)?, device, format, None)?;
    eprintln!(
        "\r{} Loaded {} ({} in memory, {:.1} s) on {}        ",
        ok_mark(),
        bold(&name),
        Bytes(model.weight_bytes() as u64),
        t0.elapsed().as_secs_f64(),
        device_label(&model.stages[0].device)
    );

    let params = SamplingParams {
        temperature: a.temperature,
        seed: a.seed,
        ..Default::default()
    };
    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(s) = &a.system {
        history.push(ChatMessage::new("system", s));
    }
    if let Some(p) = &a.prompt {
        history.push(ChatMessage::new("user", p));
        answer(&mut model, &history, &params, a.max_tokens, true)?;
        return Ok(());
    }
    println!(
        "{}",
        dim("Chat with the model. Commands: /reset, /system <text>, /exit. Ctrl-D quits.")
    );
    let stdin = std::io::stdin();
    loop {
        print!("\n{} ", bold(cyan("you ›")));
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "/exit" | "/quit" => break,
            "/reset" => {
                history.retain(|m| m.role == "system");
                println!("{}", dim("(conversation cleared)"));
                continue;
            }
            _ if line.starts_with("/system ") => {
                history.retain(|m| m.role != "system");
                history.insert(
                    0,
                    ChatMessage::new("system", line.trim_start_matches("/system ").trim()),
                );
                println!("{}", dim("(system prompt set)"));
                continue;
            }
            _ => {}
        }
        history.push(ChatMessage::new("user", line));
        print!("{} ", bold(magenta("ai  ›")));
        let reply = answer(&mut model, &history, &params, a.max_tokens, false)?;
        history.push(ChatMessage::new("assistant", &reply));
    }
    Ok(())
}

fn answer(
    model: &mut LocalModel,
    history: &[ChatMessage],
    params: &SamplingParams,
    max: usize,
    plain: bool,
) -> Result<String> {
    let prompt = model.tok.encode_chat(history)?;
    let mut out = String::new();
    let timing = model.generate(
        GenerateRequest {
            prompt,
            params: params.clone(),
            max_tokens: max,
            stop: vec![],
            ignore_eos: false,
        },
        |e| {
            if let GenEvent::Text(t) = e {
                print!("{t}");
                let _ = std::io::stdout().flush();
                out.push_str(t);
            }
            true
        },
    )?;
    println!();
    if !plain {
        println!(
            "{}",
            dim(format!(
                "      {} tokens · {:.1} tok/s · first token {:.0} ms",
                timing.completion_tokens,
                timing.decode_tps(),
                timing.ttft_ms
            ))
        );
    }
    Ok(out)
}
