//! `tendril chat` and `tendril status`: talk to a running `tendril serve`.

use crate::ui::{self, *};
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
    /// Model to talk to when the server runs several (default: its first).
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
    /// Show this model's pipeline when the server runs several.
    #[arg(long, short = 'm')]
    pub model: Option<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct LoadArgs {
    /// HuggingFace id, catalog name or a folder on the server.
    pub model: String,
    #[arg(long, default_value = "http://127.0.0.1:8080", env = "TENDRIL_URL")]
    pub url: String,
    /// Cluster token (default: the one `tendril serve` saved on this machine).
    #[arg(long, env = "TENDRIL_TOKEN")]
    pub token: Option<String>,
    /// Weight format: native, q8_0, q6_k, q4_k (default: the server's).
    #[arg(long, short = 'q')]
    pub quantize: Option<String>,
    /// Tokens per conversation, e.g. 8k (default: the server's).
    #[arg(long, short = 'c')]
    pub context: Option<String>,
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

/// `tendril load`: plan a model across the pool, show what each machine
/// gets, and load it once confirmed. Each machine then downloads only its
/// own layers.
pub fn load(a: LoadArgs) -> Result<()> {
    let token = match &a.token {
        Some(t) => t.clone(),
        None => tendril_cluster::token::load_or_create()?,
    };
    let mut req = json!({"model": a.model});
    if let Some(q) = &a.quantize {
        req["quantize"] = json!(q);
    }
    if let Some(c) = &a.context {
        let n = tendril_core::units::parse_tokens(c)
            .with_context(|| format!("cannot parse --context '{c}'"))?;
        req["context"] = json!(n);
    }
    eprintln!("{} Planning {} across the pool…", dim("·"), bold(&a.model));
    let p = control(&a.url, &token, "/api/plan", &req)?;
    print_plan(&p);
    if p["fits"] != json!(true) {
        bail!("not loading: it doesn't fit on the machines in the pool");
    }
    if !a.yes {
        print!("\n  Load it? Each machine downloads only its share. [Y/n] ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line)?;
        if line.trim().eq_ignore_ascii_case("n") || line.trim().eq_ignore_ascii_case("no") {
            println!("{}", dim("  Not loaded."));
            return Ok(());
        }
    }
    let v = control(&a.url, &token, "/api/models", &req)?;
    println!(
        "{} Loading {} — machines are downloading their shares; watch with `tendril status` or the dashboard",
        ok_mark(),
        bold(v["loaded"].as_str().unwrap_or(&a.model))
    );
    Ok(())
}

fn fmt_params(n: u64) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.1}B", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.0}M", n / 1e6)
    } else {
        format!("{:.0}K", n / 1e3)
    }
}

fn print_plan(p: &Value) {
    let b = |v: &Value| tendril_core::Bytes(v.as_u64().unwrap_or(0));
    println!();
    println!(
        "{} {} {}",
        bold("Plan ·"),
        bold(cyan(p["model"].as_str().unwrap_or(""))),
        dim(format!(
            "{} · {} params · {} layers · {} weights ({}) · {} ctx",
            p["architecture"].as_str().unwrap_or(""),
            fmt_params(p["params"].as_u64().unwrap_or(0)),
            p["layers"],
            b(&p["weight_bytes"]),
            p["format"].as_str().unwrap_or(""),
            p["context"]
        ))
    );
    let shares = p["shares"].as_array().cloned().unwrap_or_default();
    if !shares.is_empty() {
        let mut t = Table::new(&[
            "MACHINE",
            "GETS",
            "WEIGHTS",
            "SHARE",
            "MEMORY NEEDED",
            "DOWNLOADS",
        ]);
        for s in &shares {
            t.row(vec![
                bold(format!(
                    "{} {}",
                    s["machine"].as_str().unwrap_or(""),
                    dim(format!("({})", s["backend"].as_str().unwrap_or("")))
                )),
                s["components"].as_str().unwrap_or("").to_string(),
                b(&s["weight_bytes"]).to_string(),
                format!("{:.0}%", s["weight_pct"].as_f64().unwrap_or(0.0)),
                format!("{} of {}", b(&s["memory_needed"]), b(&s["memory_budget"])),
                b(&s["weight_bytes"]).to_string(),
            ]);
        }
        t.print();
        println!(
            "  {} ~{:.1} tok/s · first token ~{}",
            ok_mark(),
            p["tokens_per_sec"].as_f64().unwrap_or(0.0),
            tendril_core::units::fmt_ms(p["ttft_ms"].as_f64().unwrap_or(0.0))
        );
    } else {
        println!(
            "  {} doesn't fit: {}",
            bad_mark(),
            p["reason"].as_str().unwrap_or("not enough memory")
        );
    }
    let displaced: Vec<&str> = p["displaces"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    if !displaced.is_empty() {
        println!(
            "  {} loading it leaves no room for: {}",
            warn_mark(),
            displaced.join(", ")
        );
    }
    let fits_other: Vec<String> = p["alternatives"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter(|x| x["fits"] == json!(true))
                .map(|x| {
                    format!(
                        "{} ({}, ~{:.1} tok/s)",
                        x["format"].as_str().unwrap_or(""),
                        b(&x["weight_bytes"]),
                        x["tokens_per_sec"].as_f64().unwrap_or(0.0)
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    if !fits_other.is_empty() {
        println!(
            "  {} other formats that fit: {}  {}",
            cyan("→"),
            fits_other.join(" · "),
            dim("(--quantize <format>)")
        );
    }
    for adv in p["advice"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .take(3)
    {
        println!("  {} {}", cyan("→"), dim(adv.as_str().unwrap_or("")));
    }
}

fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .timeout(None)
        .connect_timeout(Duration::from_secs(5))
        .build()?)
}

fn fetch_status(url: &str, model: Option<&str>) -> Result<Value> {
    let mut req = client()?.get(format!("{}/api/status", url.trim_end_matches('/')));
    if let Some(m) = model {
        req = req.query(&[("model", m)]);
    }
    let r = req.send().with_context(|| {
        format!("no Tendril server at {url} — start one with `tendril serve <model>`")
    })?;
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

/// Other models the server runs, for hints.
fn other_models(st: &Value) -> Vec<String> {
    let me = st["model"].as_str().unwrap_or("");
    st["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m["id"].as_str())
                .filter(|id| *id != me)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

pub fn chat(mut a: ChatArgs) -> Result<()> {
    let st = fetch_status(&a.url, a.model.as_deref())?;
    let model = st["model"].as_str().unwrap_or("model").to_string();
    // Pin the resolved name so every request goes to the same model.
    a.model = Some(model.clone());
    let state = st["status"]["state"].as_str().unwrap_or("");
    if state != "ready" {
        println!(
            "{} {} is not ready yet ({state}). Waiting…",
            warn_mark(),
            model
        );
        loop {
            std::thread::sleep(Duration::from_secs(2));
            let s = fetch_status(&a.url, a.model.as_deref())?;
            if s["status"]["state"] == "ready" {
                break;
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
    let others = other_models(&st);
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
                let st = fetch_status(&a.url, a.model.as_deref())?;
                for m in st["models"].as_array().cloned().unwrap_or_default() {
                    let id = m["id"].as_str().unwrap_or("");
                    let mark = if Some(id) == a.model.as_deref() {
                        "›"
                    } else {
                        " "
                    };
                    println!(
                        "{} {} {}",
                        cyan(mark),
                        bold(id),
                        dim(m["state"].as_str().unwrap_or(""))
                    );
                }
                continue;
            }
            l if l.starts_with("/model ") => {
                let want = l["/model ".len()..].trim();
                match fetch_status(&a.url, Some(want)) {
                    Ok(st) => {
                        let id = st["model"].as_str().unwrap_or(want).to_string();
                        println!(
                            "{}",
                            dim(format!(
                                "(now talking to {id}; the conversation carries over)"
                            ))
                        );
                        a.model = Some(id);
                    }
                    Err(e) => println!("{} {e:#}", bad_mark()),
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
        "model": a.model.as_deref().unwrap_or("tendril"), "messages": history, "stream": true, "temperature": a.temperature,
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
        println!(
            "{}",
            dim(format!(
                "      {} tokens · {:.1} tok/s · first token {:.0} ms",
                s["usage"]["completion_tokens"],
                s["tendril"]["decode_tokens_per_sec"]
                    .as_f64()
                    .unwrap_or(0.0),
                s["tendril"]["ttft_ms"].as_f64().unwrap_or(0.0)
            ))
        );
    }
    Ok(out)
}

pub fn status(a: StatusArgs) -> Result<()> {
    let st = fetch_status(&a.url, a.model.as_deref())?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&st)?);
        return Ok(());
    }
    println!();
    println!(
        "{} {}",
        bold("Tendril ·"),
        bold(cyan(
            st["model"].as_str().unwrap_or("pool open, no model yet")
        ))
    );
    let state = st["status"]["state"].as_str().unwrap_or("?");
    let line = match state {
        "ready" => format!(
            "{} ready · {} · ~{:.1} tok/s predicted",
            ok_mark(),
            st["status"]["plan"].as_str().unwrap_or(""),
            st["status"]["predicted_tps"].as_f64().unwrap_or(0.0)
        ),
        "loading" => format!(
            "{} loading {}",
            warn_mark(),
            st["status"]["plan"].as_str().unwrap_or("")
        ),
        "waiting" => format!(
            "{} waiting: {}",
            warn_mark(),
            st["status"]["reason"].as_str().unwrap_or("")
        ),
        "failed" => format!(
            "{} failed: {}",
            bad_mark(),
            st["status"]["error"].as_str().unwrap_or("")
        ),
        other => other.to_string(),
    };
    println!("  {line}");
    let models = st["models"].as_array().cloned().unwrap_or_default();
    if models.len() > 1 {
        heading("Models");
        let mut t = Table::new(&["MODEL", "STATE", "SHARE", "ACTIVE"]);
        for m in &models {
            let share = m["machines"]
                .as_array()
                .map(|xs| {
                    xs.iter()
                        .map(|x| {
                            format!(
                                "{} {}",
                                x["name"].as_str().unwrap_or(""),
                                tendril_core::Bytes(x["budget"].as_u64().unwrap_or(0))
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" + ")
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| m["reason"].as_str().unwrap_or("–").to_string());
            let id = m["id"].as_str().unwrap_or("");
            t.row(vec![
                if id == st["model"].as_str().unwrap_or("") {
                    bold(cyan(id))
                } else {
                    bold(id)
                },
                m["state"].as_str().unwrap_or("").to_string(),
                share,
                m["active"].to_string(),
            ]);
        }
        t.print();
        println!(
            "  {}",
            dim("Below: the highlighted model. Pick another with --model <name>.")
        );
    }
    heading("Machines");
    let mut t = Table::new(&["NAME", "HARDWARE", "BACKEND", "MEMORY", "LINK", "RUNS"]);
    for n in st["nodes"].as_array().cloned().unwrap_or_default() {
        let link = match (n["rtt_ms"].as_f64(), n["bandwidth_gbps"].as_f64()) {
            _ if n["local"].as_bool() == Some(true) => dim("coordinator"),
            (Some(r), Some(b)) => format!("{} · {:.1} Gb/s", tendril_core::units::fmt_ms(r), b),
            _ => dim("measuring…"),
        };
        t.row(vec![
            bold(n["name"].as_str().unwrap_or("")),
            n["chip"].as_str().unwrap_or("").to_string(),
            n["backend"].as_str().unwrap_or("").to_string(),
            n["usable"].as_str().unwrap_or("").to_string(),
            link,
            n["role"].as_str().unwrap_or("").to_string(),
        ]);
    }
    t.print();
    let m = &st["metrics"];
    heading("Traffic");
    ui::kv(&[
        (
            "Requests",
            format!(
                "{} total · {} active · {} queued · {} errors",
                m["requests"], m["active"], m["queued"], m["errors"]
            ),
        ),
        (
            "Tokens",
            format!(
                "{} prompt · {} generated",
                m["prompt_tokens"], m["completion_tokens"]
            ),
        ),
    ]);
    println!();
    println!(
        "  {} {}",
        dim("Add a machine:"),
        cyan(st["join"].as_str().unwrap_or(""))
    );
    Ok(())
}
