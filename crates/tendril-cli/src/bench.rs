//! `tendril bench`: a repeatable measurement protocol against a server.
//!
//! Sweeps prompt lengths and concurrency, measures time-to-first-token,
//! inter-token latency (with tail percentiles), per-request and aggregate
//! throughput, compares them with the planner's predictions, and shows where
//! each token's time goes stage by stage.

use crate::ui::{self, *};
use anyhow::{bail, Context, Result};
use clap::Args;
use futures::StreamExt;
use serde::Serialize;
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tendril_core::units::fmt_ms;

#[derive(Args, Debug, Clone)]
pub struct BenchArgs {
    /// Benchmark this model in-process (starts a local server). Omit to use --url.
    pub model: Option<String>,
    /// Server to benchmark.
    #[arg(long, default_value = "http://127.0.0.1:8080", env = "TENDRIL_URL")]
    pub url: String,
    /// Concurrency levels, e.g. 1,2,4,8.
    #[arg(long, default_value = "1,2,4", value_delimiter = ',')]
    pub concurrency: Vec<usize>,
    /// Prompt lengths in tokens (approximate), e.g. 128,1024.
    #[arg(long, default_value = "128,1024", value_delimiter = ',')]
    pub prompt_tokens: Vec<usize>,
    /// Tokens generated per request.
    #[arg(long, default_value_t = 128)]
    pub output_tokens: usize,
    /// Repetitions of each configuration.
    #[arg(long, default_value_t = 3)]
    pub runs: usize,
    /// Write an HTML report here (default: tendril-bench-<time>.html).
    #[arg(long)]
    pub report: Option<std::path::PathBuf>,
    /// Skip the HTML report.
    #[arg(long)]
    pub no_report: bool,
    /// Print results as JSON.
    #[arg(long)]
    pub json: bool,
    /// Also run a multi-turn conversation of this many turns (shows prefix caching).
    #[arg(long, default_value_t = 0)]
    pub turns: usize,
    /// Weight format when benchmarking a model in-process.
    #[arg(long, short = 'q', default_value = "native")]
    pub quantize: String,
}

#[derive(Clone, Debug, Serialize)]
struct Sample {
    prompt_tokens: usize,
    cached_tokens: usize,
    text: String,
    completion_tokens: usize,
    ttft_ms: f64,
    itl_ms: Vec<f64>,
    total_ms: f64,
    decode_tps: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Row {
    prompt_target: usize,
    prompt_tokens: usize,
    concurrency: usize,
    requests: usize,
    errors: usize,
    ttft_p50: f64,
    ttft_p95: f64,
    itl_p50: f64,
    itl_p95: f64,
    itl_p99: f64,
    itl_samples: usize,
    request_tps: f64,
    aggregate_tps: f64,
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let i = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[i.min(v.len() - 1)]
}

fn make_prompt(tokens: usize, salt: usize, words_per_token: f64) -> String {
    // Vary the start so prefix caches can't cheat.
    const WORDS: &[&str] = &[
        "the",
        "river",
        "carried",
        "light",
        "across",
        "quiet",
        "fields",
        "while",
        "engineers",
        "measured",
        "latency",
        "between",
        "machines",
        "and",
        "wrote",
        "notes",
        "about",
        "memory",
        "bandwidth",
        "pipelines",
        "gardens",
        "grow",
        "slowly",
        "under",
        "patient",
        "hands",
        "every",
        "morning",
        "brings",
        "numbers",
    ];
    let n = (tokens as f64 * words_per_token).max(4.0) as usize;
    let mut s = format!("Document {salt}: ");
    for i in 0..n {
        s.push_str(WORDS[(i * 7 + salt * 13) % WORDS.len()]);
        s.push(if i % 17 == 16 { '.' } else { ' ' });
    }
    s.push_str("\nSummarize the document above in detail.");
    s
}

async fn one(
    client: &reqwest::Client,
    url: &str,
    prompt: String,
    max: usize,
    seed: u64,
) -> Result<Sample> {
    let body = json!({
        "model": "bench", "prompt": prompt, "max_tokens": max, "stream": true, "temperature": 0.7,
        "seed": seed, "ignore_eos": true, "stream_options": {"include_usage": true},
    });
    let t0 = Instant::now();
    let resp = client
        .post(format!("{url}/v1/completions"))
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        let v: Value = resp.json().await.unwrap_or(Value::Null);
        bail!(
            "{}",
            v["error"]["message"].as_str().unwrap_or("request failed")
        );
    }
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut first: Option<f64> = None;
    let mut last = 0.0;
    let mut itl = Vec::new();
    let mut usage = (0usize, 0usize);
    let mut cached = 0usize;
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        while let Some(i) = buf.find("\n\n") {
            let ev: String = buf.drain(..i + 2).collect();
            for line in ev.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    continue;
                }
                let v: Value = serde_json::from_str(data)?;
                if let Some(e) = v.get("error") {
                    bail!("{}", e["message"].as_str().unwrap_or("error"));
                }
                let now = t0.elapsed().as_secs_f64() * 1000.0;
                if let Some(t) = v["choices"][0]["text"].as_str() {
                    text.push_str(t);
                }
                if v["choices"][0]["text"]
                    .as_str()
                    .is_some_and(|t| !t.is_empty())
                {
                    match first {
                        None => first = Some(now),
                        Some(_) => itl.push(now - last),
                    }
                    last = now;
                }
                if let Some(u) = v.get("usage") {
                    usage = (
                        u["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                        u["completion_tokens"].as_u64().unwrap_or(0) as usize,
                    );
                    cached = u["prompt_tokens_details"]["cached_tokens"]
                        .as_u64()
                        .unwrap_or(0) as usize;
                }
            }
        }
    }
    let total = t0.elapsed().as_secs_f64() * 1000.0;
    let ttft = first.unwrap_or(total);
    let decode_ms = last - ttft;
    let decode_tps = if usage.1 > 1 && decode_ms > 0.0 {
        (usage.1 - 1) as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    // Chunks can merge several tokens; spread their gap evenly.
    let chunks = itl.len().max(1);
    let per = usage.1.saturating_sub(1).max(chunks) as f64 / chunks as f64;
    let itl: Vec<f64> = itl.iter().map(|g| g / per).collect();
    Ok(Sample {
        prompt_tokens: usage.0,
        cached_tokens: cached,
        text,
        completion_tokens: usage.1,
        ttft_ms: ttft,
        itl_ms: itl,
        total_ms: total,
        decode_tps,
    })
}

pub fn run(a: BenchArgs) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let (url, local) = match &a.model {
            Some(m) => {
                let (url, c) = crate::serve::start_local_for_bench(
                    m,
                    &a.quantize,
                    a.concurrency.iter().copied().max().unwrap_or(1),
                )
                .await?;
                (url, Some(c))
            }
            None => (a.url.trim_end_matches('/').to_string(), None),
        };
        let r = bench(&a, &url).await;
        if let Some(c) = local {
            c.shutdown().await;
        }
        r
    })
}

async fn bench(a: &BenchArgs, url: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .build()?;
    let status: Value = client
        .get(format!("{url}/api/status"))
        .send()
        .await
        .with_context(|| format!("no Tendril server at {url} — start one with `tendril serve <model>` or pass a model to benchmark in-process"))?
        .json()
        .await?;
    if status["status"]["state"] != "ready" {
        bail!("the server is not ready ({})", status["status"]["state"]);
    }
    let model = status["model"].as_str().unwrap_or("?").to_string();
    let plan = status["plan"].clone();
    if !a.json {
        println!();
        println!("{} {}", bold("Tendril bench ·"), bold(cyan(&model)));
        ui::kv(&[
            ("Plan", plan["label"].as_str().unwrap_or("?").to_string()),
            ("Predicted", format!("{:.1} tok/s per conversation · first token ~{}", plan["tokens_per_sec"].as_f64().unwrap_or(0.0), fmt_ms(plan["ttft_ms"].as_f64().unwrap_or(0.0)))),
            ("Protocol", format!("prompts {:?} tokens × concurrency {:?} × {} runs · {} output tokens each (EOS ignored)", a.prompt_tokens, a.concurrency, a.runs, a.output_tokens)),
        ]);
        println!();
        print!("  {} warming up… ", dim("·"));
    }
    // Learn this tokenizer's words-per-token so prompt lengths hit their targets.
    let mut wpt = 0.75;
    let probe = make_prompt(400, 1, wpt);
    if let Ok(r) = client
        .post(format!("{url}/tokenize"))
        .json(&json!({"prompt": probe}))
        .send()
        .await
    {
        if let Ok(v) = r.json::<Value>().await {
            if let Some(n) = v["count"].as_f64().filter(|n| *n > 0.0) {
                wpt = (wpt * 400.0 / n).clamp(0.05, 4.0);
            }
        }
    }
    one(&client, url, make_prompt(32, 999, wpt), 8, 1)
        .await
        .context("warm-up request failed")?;
    if !a.json {
        println!("{}", ok_mark());
    }
    let mut rows = Vec::new();
    let mut all_samples = Vec::new();
    let mut salt = 0usize;
    for &p in &a.prompt_tokens {
        for &c in &a.concurrency {
            let mut samples = Vec::new();
            let mut errors = 0;
            let mut agg = Vec::new();
            for _ in 0..a.runs {
                let t0 = Instant::now();
                let futs: Vec<_> = (0..c)
                    .map(|_| {
                        salt += 1;
                        one(
                            &client,
                            url,
                            make_prompt(p, salt, wpt),
                            a.output_tokens,
                            salt as u64,
                        )
                    })
                    .collect();
                let res = futures::future::join_all(futs).await;
                let wall = t0.elapsed().as_secs_f64();
                let mut toks = 0;
                for r in res {
                    match r {
                        Ok(s) => {
                            toks += s.completion_tokens;
                            samples.push(s);
                        }
                        Err(e) => {
                            errors += 1;
                            if !a.json {
                                println!("  {} {e:#}", bad_mark());
                            }
                        }
                    }
                }
                agg.push(toks as f64 / wall);
            }
            let mut ttft: Vec<f64> = samples.iter().map(|s| s.ttft_ms).collect();
            let mut itl: Vec<f64> = samples
                .iter()
                .flat_map(|s| s.itl_ms.iter().copied())
                .collect();
            let req_tps =
                samples.iter().map(|s| s.decode_tps).sum::<f64>() / samples.len().max(1) as f64;
            let row = Row {
                prompt_target: p,
                prompt_tokens: samples.iter().map(|s| s.prompt_tokens).sum::<usize>()
                    / samples.len().max(1),
                concurrency: c,
                requests: samples.len(),
                errors,
                ttft_p50: pct(&mut ttft, 0.5),
                ttft_p95: pct(&mut ttft, 0.95),
                itl_p50: pct(&mut itl, 0.5),
                itl_p95: pct(&mut itl, 0.95),
                itl_p99: pct(&mut itl, 0.99),
                itl_samples: itl.len(),
                request_tps: req_tps,
                aggregate_tps: agg.iter().sum::<f64>() / agg.len().max(1) as f64,
            };
            if !a.json {
                println!(
                    "  {} prompt ~{:>5} tok · concurrency {:>2} · {:>6.1} tok/s total · {:>6.1} tok/s each · first token {}",
                    ok_mark(),
                    row.prompt_tokens,
                    c,
                    row.aggregate_tps,
                    row.request_tps,
                    fmt_ms(row.ttft_p50)
                );
            }
            all_samples.extend(samples);
            rows.push(row);
        }
    }
    let mut conversation = Vec::new();
    if a.turns > 0 {
        if !a.json {
            heading("Multi-turn conversation");
        }
        let mut history = make_prompt(a.prompt_tokens.first().copied().unwrap_or(256), 4242, wpt);
        for turn in 1..=a.turns {
            let s = one(
                &client,
                url,
                history.clone(),
                a.output_tokens,
                7 + turn as u64,
            )
            .await?;
            if !a.json {
                println!(
                    "  {} turn {turn}: prompt {:>5} tok · {:>5} reused from cache · first token {}",
                    ok_mark(),
                    s.prompt_tokens,
                    s.cached_tokens,
                    fmt_ms(s.ttft_ms)
                );
            }
            history.push_str(&s.text);
            history.push_str(&format!(
                "\nUser: tell me more about part {turn}.\nAssistant:"
            ));
            conversation.push(json!({"turn": turn, "prompt_tokens": s.prompt_tokens, "cached_tokens": s.cached_tokens, "ttft_ms": s.ttft_ms}));
        }
    }
    let after: Value = client
        .get(format!("{url}/api/status"))
        .send()
        .await?
        .json()
        .await?;
    let tel = after["telemetry"].clone();

    if a.json {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &json!({"model": model, "plan": plan, "rows": rows, "telemetry": tel, "conversation": conversation})
            )?
        );
        return Ok(());
    }

    heading("Results");
    let mut t = Table::new(&[
        "PROMPT",
        "CONC",
        "REQS",
        "FIRST TOKEN p50/p95",
        "INTER-TOKEN p50/p95/p99",
        "TOK/S EACH",
        "TOK/S TOTAL",
        "ERR",
    ])
    .right(&[0, 1, 2, 3, 4, 5, 6, 7]);
    for r in &rows {
        t.row(vec![
            r.prompt_tokens.to_string(),
            r.concurrency.to_string(),
            r.requests.to_string(),
            format!("{} / {}", fmt_ms(r.ttft_p50), fmt_ms(r.ttft_p95)),
            format!(
                "{} / {} / {}",
                fmt_ms(r.itl_p50),
                fmt_ms(r.itl_p95),
                fmt_ms(r.itl_p99)
            ),
            format!("{:.1}", r.request_tps),
            bold(format!("{:.1}", r.aggregate_tps)),
            if r.errors > 0 {
                red(r.errors)
            } else {
                dim("0")
            },
        ]);
    }
    t.print();
    println!("  {}", dim("p99 needs many samples to be meaningful; raise --runs or --output-tokens for tighter tails."));

    // Prediction vs measurement.
    let pred_tps = plan["tokens_per_sec"].as_f64().unwrap_or(0.0);
    if let Some(r1) = rows.iter().find(|r| r.concurrency == 1) {
        heading("Planner prediction vs measured");
        let err = (pred_tps - r1.request_tps) / r1.request_tps.max(1e-9) * 100.0;
        ui::kv(&[(
            "Decode speed",
            format!(
                "predicted {:.1} tok/s · measured {:.1} tok/s ({:+.0}%)",
                pred_tps, r1.request_tps, err
            ),
        )]);
    }
    if tel.is_object() {
        heading("Where each token's time goes (server telemetry)");
        let mut t =
            Table::new(&["STAGE", "PREDICTED", "MEASURED COMPUTE", "QUEUED"]).right(&[1, 2, 3]);
        for s in tel["stages"].as_array().cloned().unwrap_or_default() {
            t.row(vec![
                s["node"].as_str().unwrap_or("").to_string(),
                fmt_ms(s["predicted_ms"].as_f64().unwrap_or(0.0)),
                fmt_ms(s["compute_ms"].as_f64().unwrap_or(0.0)),
                fmt_ms(s["queue_ms"].as_f64().unwrap_or(0.0)),
                format!("{:.1}", s["batch"].as_f64().unwrap_or(1.0)),
            ]);
        }
        t.row(vec![
            "network + scheduling".into(),
            fmt_ms(tel["predicted_network_ms"].as_f64().unwrap_or(0.0)),
            fmt_ms(tel["transfer_ms"].as_f64().unwrap_or(0.0)),
            String::new(),
        ]);
        t.row(vec![
            bold("per token"),
            bold(fmt_ms(tel["predicted_step_ms"].as_f64().unwrap_or(0.0))),
            bold(fmt_ms(tel["step_ms"].as_f64().unwrap_or(0.0))),
            String::new(),
        ]);
        t.print();
        println!("  {}", dim("The server feeds these measurements back into future plans (see `Calibrated:` events)."));
    }
    if !a.no_report {
        let path = a.report.clone().unwrap_or_else(|| {
            std::path::PathBuf::from(format!("tendril-bench-{}.html", chrono_stamp()))
        });
        std::fs::write(
            &path,
            crate::bench_report::html(&model, &plan, &rows, &tel, &all_samples_itl(&all_samples)),
        )?;
        println!();
        println!("{} Report: {}", ok_mark(), bold(path.display()));
    }
    Ok(())
}

fn all_samples_itl(s: &[Sample]) -> Vec<f64> {
    s.iter().flat_map(|x| x.itl_ms.iter().copied()).collect()
}

fn chrono_stamp() -> String {
    tendril_cluster::coordinator::clock().replace(':', "")
}
