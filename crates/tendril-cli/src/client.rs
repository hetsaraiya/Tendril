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
        bold(cyan(st["model"].as_str().unwrap_or("")))
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
