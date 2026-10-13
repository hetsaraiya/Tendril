//! `tendril chat` and `tendril status`: talk to a running `tendril serve`.

use crate::ui::*;
use anyhow::{bail, Context, Result};
use clap::Args;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::time::Duration;

#[derive(Args, Debug)]
pub struct ChatArgs {
    /// Server URL.
    #[arg(long, default_value = "http://127.0.0.1:8080", env = "TENDRIL_URL")]
    pub url: String,
    /// Model to talk to when the server runs several (default: the first chat model).
    #[arg(long, short = 'm')]
    pub model: Option<String>,
    /// System prompt.
    #[arg(long)]
    pub system: Option<String>,
    /// Send one message and exit.
    #[arg(long, short = 'p')]
    pub prompt: Option<String>,
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,
    #[arg(long, default_value_t = 1024)]
    pub max_tokens: usize,
}

#[derive(Args, Debug)]
pub struct StatusArgs {
    #[arg(long, default_value = "http://127.0.0.1:8080", env = "TENDRIL_URL")]
    pub url: String,
    /// Show only this model.
    #[arg(long, short = 'm')]
    pub model: Option<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct LoadArgs {
    /// The model, in the engine's terms (`tendril engines` shows examples).
    pub model: String,
    /// Engine to run it with: llama.cpp, vllm, sglang, ollama, mlx, exo,
    /// comfyui, sd.cpp, whisper.cpp.
    #[arg(long, short = 'e', default_value = "llama.cpp")]
    pub engine: String,
    #[arg(long, default_value = "http://127.0.0.1:8080", env = "TENDRIL_URL")]
    pub url: String,
    /// Cluster token (default: the one `tendril serve` saved on this machine).
    #[arg(long, env = "TENDRIL_TOKEN")]
    pub token: Option<String>,
    /// Quantization (engine-specific, e.g. Q4_K_M for llama.cpp).
    #[arg(long, short = 'q')]
    pub quantize: Option<String>,
    /// Tokens per conversation, e.g. 8k (default: the server's).
    #[arg(long, short = 'c')]
    pub context: Option<String>,
    /// Conversations at once (default: the server's).
    #[arg(long)]
    pub concurrency: Option<usize>,
    /// Only use these machines (names, comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub machines: Option<Vec<String>>,
    /// Split across at least this many machines.
    #[arg(long)]
    pub min_machines: Option<usize>,
    /// The machines share a private network you trust (needed to split a
    /// model with engines whose own traffic isn't encrypted).
    #[arg(long)]
    pub trusted_network: bool,
    /// Load without asking.
    #[arg(long, short = 'y')]
    pub yes: bool,
}

/// POST `path` with the cluster token; the server's error message on failure.
fn control(url: &str, token: &str, path: &str, body: &Value) -> Result<Value> {
    let r = client()?
        .post(format!("{}{path}", url.trim_end_matches('/')))
        .bearer_auth(token)
        .json(body)
        .send()
        .with_context(|| format!("no Tendril server at {url} — start one with `tendril serve`"))?;
    let ok = r.status().is_success();
    let v: Value = r.json()?;
    if !ok {
        bail!(
            "{}",
            v["error"]["message"].as_str().unwrap_or("request failed")
        );
    }
    Ok(v)
}

fn ask(q: &str) -> Result<bool> {
    print!("{q} [Y/n] ");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    let l = line.trim().to_ascii_lowercase();
    Ok(!(l == "n" || l == "no"))
}

/// `tendril load`: plan a model with an engine, show what each machine
/// gets and what it needs, and load it once confirmed.
pub fn load(a: LoadArgs) -> Result<()> {
    let token = match &a.token {
        Some(t) => t.clone(),
        None => tendril_cluster::token::load_or_create()?,
    };
    let mut req =
        json!({"model": a.model, "engine": a.engine, "trusted_network": a.trusted_network});
    if let Some(q) = &a.quantize {
        req["quantize"] = json!(q);
    }
    if let Some(c) = &a.context {
        let n = tendril_core::units::parse_tokens(c)
            .with_context(|| format!("cannot parse --context '{c}'"))?;
        req["context"] = json!(n);
    }
    if let Some(n) = a.concurrency {
        req["concurrency"] = json!(n);
    }
    if let Some(m) = &a.machines {
        req["machines"] = json!(m);
    }
    if let Some(n) = a.min_machines {
        req["min_machines"] = json!(n);
    }
    eprintln!(
        "{} Planning {} with {}…",
        dim("·"),
        bold(&a.model),
        bold(&a.engine)
    );
    let v = loop {
        let p = control(&a.url, &token, "/api/plan", &req)?;
        print_plan(&p);
        if p["fits"] != json!(true) {
            bail!(
                "not loading: {}",
                p["reason"]
                    .as_str()
                    .unwrap_or("it can't run on these machines")
            );
        }
        if p["needs_trusted_network"] == json!(true) && req["trusted_network"] != json!(true) {
            println!();
            println!(
                "  {} {}",
                warn_mark(),
                bold("This split sends the engine's traffic between machines unencrypted.")
            );
            println!("    {}", p["risk"].as_str().unwrap_or(""));
            if a.yes || !ask("  Are these machines on a private network you trust?")? {
                bail!("not loading: confirm a trusted network with --trusted-network, or limit it to one machine with --machines");
            }
            req["trusted_network"] = json!(true);
        }
        if !a.yes && !ask("\n  Load it?")? {
            println!("{}", dim("  Not loaded."));
            return Ok(());
        }
        let mut confirmed = req.clone();
        confirmed["expect_version"] = p["pool_version"].clone();
        match control(&a.url, &token, "/api/models", &confirmed) {
            Ok(v) => break v,
            Err(e) if e.to_string().contains("plan again") && !a.yes => {
                println!(
                    "\n{} The machines changed while you were deciding — here is the new plan:",
                    warn_mark()
                );
            }
            Err(e) => return Err(e),
        }
    };
    println!(
        "{} Loading {} — watch with `tendril status` or the dashboard",
        ok_mark(),
        bold(v["loading"].as_str().unwrap_or(&a.model))
    );
    Ok(())
}

fn print_plan(p: &Value) {
    let b = |v: &Value| tendril_core::Bytes(v.as_u64().unwrap_or(0));
    println!();
    println!(
        "{} {} {}",
        bold("Plan ·"),
        bold(cyan(p["model"].as_str().unwrap_or(""))),
        dim(format!(
            "with {} · {} · {} layers · {} · {} ctx × {}",
            p["engine"]["name"].as_str().unwrap_or(""),
            p["task"].as_str().unwrap_or(""),
            p["layers"],
            b(&p["weight_bytes"]),
            p["context"],
            p["concurrency"]
        ))
    );
    for n in p["notes"].as_array().cloned().unwrap_or_default() {
        println!("  {} {}", dim("·"), dim(n.as_str().unwrap_or("")));
    }
    let stages = p["stages"].as_array().cloned().unwrap_or_default();
    if !stages.is_empty() {
        let mut t = Table::new(&["MACHINE", "LAYERS", "WEIGHTS", "SHARE", "MEMORY", "MOVES"]);
        for s in &stages {
            let l = &s["layers"];
            let layers = if l[1].as_u64().unwrap_or(0) > 0 {
                format!("{}–{}", l[0], l[1].as_u64().unwrap_or(1) - 1)
            } else {
                "–".into()
            };
            t.row(vec![
                bold(format!(
                    "{}{} {}",
                    s["machine"].as_str().unwrap_or(""),
                    if s["serves_api"] == json!(true) {
                        " ★"
                    } else {
                        ""
                    },
                    dim(format!("({})", s["backend"].as_str().unwrap_or("")))
                )),
                layers,
                b(&s["weight_bytes"]).to_string(),
                format!("{:.0}%", s["weight_pct"].as_f64().unwrap_or(0.0)),
                format!(
                    "{} of {} free",
                    b(&s["memory_needed"]),
                    b(&s["memory_free"])
                ),
                dim(s["transfer"].as_str().unwrap_or("")),
            ]);
        }
        t.print();
        if let Some(tps) = p["tokens_per_sec"].as_f64() {
            println!(
                "  {} ~{:.1} tok/s predicted · ★ serves the API",
                ok_mark(),
                tps
            );
        }
        println!(
            "  {} {}",
            dim("Network:"),
            dim(p["security"].as_str().unwrap_or(""))
        );
    } else {
        println!(
            "  {} {}",
            bad_mark(),
            p["reason"].as_str().unwrap_or("not enough memory")
        );
    }
    for u in p["unavailable"].as_array().cloned().unwrap_or_default() {
        let missing: Vec<String> = u["missing"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .map(|m| {
                format!(
                    "{}{}",
                    m["name"].as_str().unwrap_or(""),
                    m["install"]
                        .as_str()
                        .map(|i| format!(" ({i})"))
                        .unwrap_or_default()
                )
            })
            .collect();
        println!(
            "  {} {} can't run {}: missing {}",
            warn_mark(),
            bold(u["machine"].as_str().unwrap_or("")),
            p["engine"]["name"].as_str().unwrap_or(""),
            missing.join("; ")
        );
    }
}

fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .timeout(None)
        .connect_timeout(Duration::from_secs(5))
        .build()?)
}

fn fetch_status(url: &str) -> Result<Value> {
    let r = client()?
        .get(format!("{}/api/status", url.trim_end_matches('/')))
        .send()
        .with_context(|| format!("no Tendril server at {url} — start one with `tendril serve`"))?;
    let ok = r.status().is_success();
    let v: Value = r.json()?;
    if !ok {
        bail!(
            "{}",
            v["error"]["message"].as_str().unwrap_or("request failed")
        );
    }
    Ok(v)
}

/// Chat models the server runs: (name, status).
fn chat_models(st: &Value) -> Vec<(String, String)> {
    st["deployments"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter(|d| d["task"] == json!("chat"))
        .map(|d| {
            (
                d["name"].as_str().unwrap_or("").to_string(),
                d["status"].as_str().unwrap_or("").to_string(),
            )
        })
        .collect()
}

pub fn chat(mut a: ChatArgs) -> Result<()> {
    let st = fetch_status(&a.url)?;
    let models = chat_models(&st);
    let model = match &a.model {
        Some(m) => m.clone(),
        None => match models.first() {
            Some((m, _)) => m.clone(),
            None => bail!("no chat model is loaded — `tendril load <model> --engine <engine>`"),
        },
    };
    a.model = Some(model.clone());
    let state = |st: &Value| {
        chat_models(st)
            .into_iter()
            .find(|(n, _)| *n == model)
            .map(|(_, s)| s)
            .unwrap_or_default()
    };
    if state(&st) != "ready" {
        println!(
            "{} {} is not ready yet ({}). Waiting…",
            warn_mark(),
            model,
            state(&st)
        );
        loop {
            std::thread::sleep(Duration::from_secs(2));
            let s = state(&fetch_status(&a.url)?);
            if s == "ready" {
                break;
            }
            if s == "failed" || s.is_empty() {
                bail!("{model} failed to load — see `tendril status`");
            }
        }
    }
    println!(
        "{} {} {}",
        bold("Chatting with"),
        bold(cyan(&model)),
        dim(format!("via {}", a.url))
    );
    let mut history: Vec<Value> = Vec::new();
    if let Some(s) = &a.system {
        history.push(json!({"role": "system", "content": s}));
    }
    if let Some(p) = &a.prompt {
        history.push(json!({"role": "user", "content": p}));
        send(&a, &history, true)?;
        return Ok(());
    }
    let others: Vec<String> = models
        .iter()
        .map(|(n, _)| n.clone())
        .filter(|n| *n != model)
        .collect();
    if !others.is_empty() {
        println!(
            "{}",
            dim(format!(
                "Also serving {} — switch with /model <name>.",
                others.join(", ")
            ))
        );
    }
    println!(
        "{}",
        dim("Commands: /reset, /model <name>, /exit. Ctrl-D quits.")
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
        match line {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                history.retain(|m| m["role"] == "system");
                println!("{}", dim("(conversation cleared)"));
                continue;
            }
            "/model" | "/models" => {
                for (id, state) in chat_models(&fetch_status(&a.url)?) {
                    let mark = if Some(id.as_str()) == a.model.as_deref() {
                        "›"
                    } else {
                        " "
                    };
                    println!("{} {} {}", cyan(mark), bold(&id), dim(state));
                }
                continue;
            }
            l if l.starts_with("/model ") => {
                let want = l["/model ".len()..].trim().to_string();
                if chat_models(&fetch_status(&a.url)?)
                    .iter()
                    .any(|(n, _)| *n == want)
                {
                    println!(
                        "{}",
                        dim(format!(
                            "(now talking to {want}; the conversation carries over)"
                        ))
                    );
                    a.model = Some(want);
                } else {
                    println!("{} no chat model named {want}", bad_mark());
                }
                continue;
            }
            _ => {}
        }
        history.push(json!({"role": "user", "content": line}));
        print!("{} ", bold(magenta("ai  ›")));
        match send(&a, &history, false) {
            Ok(reply) => history.push(json!({"role": "assistant", "content": reply})),
            Err(e) => {
                println!("\n{} {e:#}", bad_mark());
                history.pop();
            }
        }
    }
    Ok(())
}

fn send(a: &ChatArgs, history: &[Value], plain: bool) -> Result<String> {
    let body = json!({
        "model": a.model.as_deref().unwrap_or(""), "messages": history, "stream": true, "temperature": a.temperature,
        "max_tokens": a.max_tokens, "stream_options": {"include_usage": true},
    });
    let resp = client()?
        .post(format!(
            "{}/v1/chat/completions",
            a.url.trim_end_matches('/')
        ))
        .json(&body)
        .send()?;
    if !resp.status().is_success() {
        let v: Value = resp.json().unwrap_or(Value::Null);
        bail!(
            "{}",
            v["error"]["message"].as_str().unwrap_or("request failed")
        );
    }
    let t0 = std::time::Instant::now();
    let mut out = String::new();
    let mut stats = None;
    for line in BufReader::new(resp).lines() {
        let line = line?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let v: Value = serde_json::from_str(data)?;
        if let Some(e) = v.get("error") {
            bail!("{}", e["message"].as_str().unwrap_or("error"));
        }
        if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
            print!("{t}");
            std::io::stdout().flush()?;
            out.push_str(t);
        }
        if v.get("usage").is_some() {
            stats = Some(v);
        }
    }
    println!();
    if let (Some(s), false) = (stats, plain) {
        let secs = t0.elapsed().as_secs_f64();
        let n = s["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        println!(
            "{}",
            dim(format!(
                "      {n} tokens · {:.1} tok/s",
                n as f64 / secs.max(1e-3)
            ))
        );
    }
    Ok(out)
}

pub fn status(a: StatusArgs) -> Result<()> {
    let st = fetch_status(&a.url)?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&st)?);
        return Ok(());
    }
    println!();
    println!(
        "{} {}",
        bold("Tendril · cluster"),
        bold(cyan(st["cluster"].as_str().unwrap_or("")))
    );
    heading("Models");
    let deps = st["deployments"].as_array().cloned().unwrap_or_default();
    if deps.is_empty() {
        println!(
            "  {}",
            dim("none loaded — `tendril load <model> --engine <engine>` or the dashboard")
        );
    }
    for d in &deps {
        if a.model.as_deref().is_some_and(|m| d["name"] != json!(m)) {
            continue;
        }
        let status = d["status"].as_str().unwrap_or("");
        let mark = match status {
            "ready" => ok_mark(),
            "failed" => bad_mark(),
            _ => warn_mark(),
        };
        println!(
            "  {mark} {} {} {}",
            bold(d["name"].as_str().unwrap_or("")),
            dim(format!(
                "({}, {})",
                d["engine_name"].as_str().unwrap_or(""),
                d["task"].as_str().unwrap_or("")
            )),
            match status {
                "failed" => red(d["reason"].as_str().unwrap_or("failed")),
                s => s.to_string(),
            }
        );
        for r in d["roles"].as_array().cloned().unwrap_or_default() {
            let (done, total) = (
                r["done"].as_u64().unwrap_or(0),
                r["total"].as_u64().unwrap_or(0),
            );
            let prog = if total > 0 {
                format!(
                    " {}/{}",
                    tendril_core::Bytes(done),
                    tendril_core::Bytes(total)
                )
            } else {
                String::new()
            };
            println!(
                "      {} on {}: {}{}{}",
                r["id"].as_str().unwrap_or(""),
                r["machine_name"].as_str().unwrap_or(""),
                r["state"].as_str().unwrap_or(""),
                r["detail"]
                    .as_str()
                    .filter(|x| !x.is_empty())
                    .map(|x| format!(" · {x}"))
                    .unwrap_or_default(),
                prog
            );
        }
    }
    heading("Machines");
    let mut t = Table::new(&["NAME", "HARDWARE", "MEMORY FREE", "ENGINES", "LINK"]);
    for m in st["machines"].as_array().cloned().unwrap_or_default() {
        let engines: Vec<String> = m["engines"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter(|e| e["ready"] == json!(true))
            .map(|e| e["engine"].as_str().unwrap_or("").to_string())
            .collect();
        t.row(vec![
            bold(format!(
                "{}{}",
                m["name"].as_str().unwrap_or(""),
                if m["local"] == json!(true) {
                    " (server)"
                } else {
                    ""
                }
            )),
            format!(
                "{} · {}",
                m["chip"].as_str().unwrap_or(""),
                m["backend"].as_str().unwrap_or("")
            ),
            format!(
                "{} of {}",
                tendril_core::Bytes(m["free_memory"].as_u64().unwrap_or(0)),
                tendril_core::Bytes(m["usable_memory"].as_u64().unwrap_or(0))
            ),
            if engines.is_empty() {
                yellow("none")
            } else {
                engines.join(", ")
            },
            m["rtt_ms"]
                .as_f64()
                .map(tendril_core::units::fmt_ms)
                .unwrap_or_else(|| dim("–")),
        ]);
    }
    t.print();
    println!();
    println!(
        "  {} {}",
        dim("Add a machine:"),
        cyan(st["join"].as_str().unwrap_or(""))
    );
    Ok(())
}
