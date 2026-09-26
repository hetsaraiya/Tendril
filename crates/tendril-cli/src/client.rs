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
    #[arg(long)]
    pub json: bool,
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
        .with_context(|| {
            format!("no Tendril server at {url} — start one with `tendril serve <model>`")
        })?;
    Ok(r.json()?)
}

pub fn chat(a: ChatArgs) -> Result<()> {
    let st = fetch_status(&a.url)?;
    let model = st["model"].as_str().unwrap_or("model").to_string();
    let state = st["status"]["state"].as_str().unwrap_or("");
    if state != "ready" {
        println!(
            "{} {} is not ready yet ({state}). Waiting…",
            warn_mark(),
            model
        );
        loop {
            std::thread::sleep(Duration::from_secs(2));
            let s = fetch_status(&a.url)?;
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
    println!("{}", dim("Commands: /reset, /exit. Ctrl-D quits."));
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
        "model": "tendril", "messages": history, "stream": true, "temperature": a.temperature,
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
    let st = fetch_status(&a.url)?;
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
