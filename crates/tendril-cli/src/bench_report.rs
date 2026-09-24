//! Self-contained HTML report for `tendril bench` (inline SVG, no assets).

use crate::bench::Row;
use serde_json::Value;

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn ms(v: f64) -> String {
    tendril_core::units::fmt_ms(v)
}

const PALETTE: &[&str] = &[
    "#16a34a", "#2563eb", "#d97706", "#9333ea", "#dc2626", "#0891b2",
];

/// A small line chart: series of (label, points (x, y)).
fn line_chart(
    title: &str,
    xlabel: &str,
    ylabel: &str,
    series: &[(String, Vec<(f64, f64)>)],
    yfmt: fn(f64) -> String,
) -> String {
    let (w, h, l, r, t, b) = (560.0, 260.0, 56.0, 16.0, 16.0, 40.0);
    let xs: Vec<f64> = series
        .iter()
        .flat_map(|s| s.1.iter().map(|p| p.0))
        .collect();
    let ys: Vec<f64> = series
        .iter()
        .flat_map(|s| s.1.iter().map(|p| p.1))
        .collect();
    if xs.is_empty() {
        return String::new();
    }
    let (x0, x1) = (
        xs.iter().cloned().fold(f64::INFINITY, f64::min),
        xs.iter().cloned().fold(0.0, f64::max),
    );
    let y1 = ys.iter().cloned().fold(0.0, f64::max).max(1e-9) * 1.15;
    let sx = |x: f64| {
        if x1 > x0 {
            l + (x - x0) / (x1 - x0) * (w - l - r)
        } else {
            (l + w - r) / 2.0
        }
    };
    let sy = |y: f64| h - b - y / y1 * (h - t - b);
    let mut svg = format!(
        r#"<svg viewBox="0 0 {w} {h}" class="chart"><g font-size="11" fill="currentColor">"#
    );
    for i in 0..=4 {
        let v = y1 * i as f64 / 4.0;
        let y = sy(v);
        svg += &format!(
            r#"<line x1="{l}" x2="{}" y1="{y}" y2="{y}" stroke="currentColor" opacity=".12"/><text x="{}" y="{}" text-anchor="end" opacity=".7">{}</text>"#,
            w - r,
            l - 6.0,
            y + 4.0,
            yfmt(v)
        );
    }
    let mut xticks: Vec<f64> = xs.clone();
    xticks.sort_by(|a, b| a.total_cmp(b));
    xticks.dedup();
    for x in &xticks {
        svg += &format!(
            r#"<text x="{}" y="{}" text-anchor="middle" opacity=".7">{}</text>"#,
            sx(*x),
            h - b + 16.0,
            x
        );
    }
    svg += &format!(
        r#"<text x="{}" y="{}" text-anchor="middle" opacity=".7">{}</text>"#,
        (l + w - r) / 2.0,
        h - 6.0,
        esc(xlabel)
    );
    svg += &format!(
        r#"<text x="12" y="{}" transform="rotate(-90 12 {})" text-anchor="middle" opacity=".7">{}</text>"#,
        h / 2.0,
        h / 2.0,
        esc(ylabel)
    );
    for (i, (name, pts)) in series.iter().enumerate() {
        let c = PALETTE[i % PALETTE.len()];
        let d: Vec<String> = pts
            .iter()
            .map(|(x, y)| format!("{:.1},{:.1}", sx(*x), sy(*y)))
            .collect();
        svg += &format!(
            r#"<polyline points="{}" fill="none" stroke="{c}" stroke-width="2.2"/>"#,
            d.join(" ")
        );
        for (x, y) in pts {
            svg += &format!(
                r#"<circle cx="{:.1}" cy="{:.1}" r="3.5" fill="{c}"/>"#,
                sx(*x),
                sy(*y)
            );
        }
        svg += &format!(
            r#"<rect x="{}" y="{}" width="10" height="10" rx="2" fill="{c}"/><text x="{}" y="{}">{}</text>"#,
            l + 8.0 + i as f64 * 150.0,
            t,
            l + 22.0 + i as f64 * 150.0,
            t + 9.0,
            esc(name)
        );
    }
    svg += "</g></svg>";
    format!("<section><h2>{}</h2>{svg}</section>", esc(title))
}

fn histogram(title: &str, values: &[f64]) -> String {
    if values.len() < 2 {
        return String::new();
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let hi = v[((v.len() as f64 - 1.0) * 0.99) as usize].max(1e-6);
    let bins = 30usize;
    let mut counts = vec![0usize; bins];
    for x in &v {
        let i = ((x / hi) * bins as f64).floor() as usize;
        counts[i.min(bins - 1)] += 1;
    }
    let maxc = *counts.iter().max().unwrap_or(&1) as f64;
    let (w, h, l, b) = (560.0, 200.0, 16.0, 34.0);
    let bw = (w - 2.0 * l) / bins as f64;
    let mut svg = format!(
        r#"<svg viewBox="0 0 {w} {h}" class="chart"><g font-size="11" fill="currentColor">"#
    );
    for (i, c) in counts.iter().enumerate() {
        let bh = *c as f64 / maxc * (h - b - 12.0);
        svg += &format!(
            r##"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" rx="1.5" fill="#16a34a" opacity=".85"/>"##,
            l + i as f64 * bw + 1.0,
            h - b - bh,
            bw - 2.0,
            bh
        );
    }
    for i in 0..=4 {
        let x = l + (w - 2.0 * l) * i as f64 / 4.0;
        svg += &format!(
            r#"<text x="{x:.1}" y="{}" text-anchor="middle" opacity=".7">{}</text>"#,
            h - b + 16.0,
            ms(hi * i as f64 / 4.0)
        );
    }
    svg += "</g></svg>";
    format!("<section><h2>{}</h2>{svg}<p class=\"sub\">{} gaps between streamed tokens (99th percentile range shown).</p></section>", esc(title), v.len())
}

fn breakdown(tel: &Value) -> String {
    let Some(stages) = tel["stages"].as_array() else {
        return String::new();
    };
    let mut rows = String::new();
    let max = stages
        .iter()
        .map(|s| {
            s["predicted_ms"]
                .as_f64()
                .unwrap_or(0.0)
                .max(s["compute_ms"].as_f64().unwrap_or(0.0))
        })
        .fold(
            tel["transfer_ms"]
                .as_f64()
                .unwrap_or(0.0)
                .max(tel["predicted_network_ms"].as_f64().unwrap_or(0.0)),
            f64::max,
        )
        .max(1e-9);
    let mut row = |name: &str, pred: f64, meas: f64| {
        rows += &format!(
            r#"<div class="bd"><div class="bn">{}</div><div class="bbars"><div class="bar p" style="width:{:.1}%"></div><span>{}</span><div class="bar m" style="width:{:.1}%"></div><span>{}</span></div></div>"#,
            esc(name),
            pred / max * 100.0,
            ms(pred),
            meas / max * 100.0,
            ms(meas)
        );
    };
    for s in stages {
        row(
            s["node"].as_str().unwrap_or(""),
            s["predicted_ms"].as_f64().unwrap_or(0.0),
            s["compute_ms"].as_f64().unwrap_or(0.0),
        );
    }
    row(
        "network + scheduling",
        tel["predicted_network_ms"].as_f64().unwrap_or(0.0),
        tel["transfer_ms"].as_f64().unwrap_or(0.0),
    );
    format!(
        r#"<section><h2>Where each token's time goes</h2><div class="legend"><span class="sw p"></span>predicted <span class="sw m"></span>measured</div>{rows}<p class="sub">Per decode step: {} predicted, {} measured over {} steps.</p></section>"#,
        ms(tel["predicted_step_ms"].as_f64().unwrap_or(0.0)),
        ms(tel["step_ms"].as_f64().unwrap_or(0.0)),
        tel["samples"]
    )
}

pub fn html(model: &str, plan: &Value, rows: &[Row], tel: &Value, itl: &[f64]) -> String {
    let rows_json: Vec<Value> = rows
        .iter()
        .map(|r| serde_json::to_value(r).unwrap())
        .collect();
    let mut prompts: Vec<usize> = rows_json
        .iter()
        .map(|r| r["prompt_target"].as_u64().unwrap_or(0) as usize)
        .collect();
    prompts.sort();
    prompts.dedup();
    let mut concs: Vec<usize> = rows_json
        .iter()
        .map(|r| r["concurrency"].as_u64().unwrap_or(0) as usize)
        .collect();
    concs.sort();
    concs.dedup();
    let tput: Vec<(String, Vec<(f64, f64)>)> = prompts
        .iter()
        .map(|p| {
            (
                format!("~{p}-token prompts"),
                rows_json
                    .iter()
                    .filter(|r| r["prompt_target"] == *p)
                    .map(|r| {
                        (
                            r["concurrency"].as_f64().unwrap_or(0.0),
                            r["aggregate_tps"].as_f64().unwrap_or(0.0),
                        )
                    })
                    .collect(),
            )
        })
        .collect();
    let ttft: Vec<(String, Vec<(f64, f64)>)> = concs
        .iter()
        .map(|c| {
            (
                format!("concurrency {c}"),
                rows_json
                    .iter()
                    .filter(|r| r["concurrency"] == *c)
                    .map(|r| {
                        let t = &r["prompt_target"];
                        let same: Vec<f64> = rows_json
                            .iter()
                            .filter(|x| &x["prompt_target"] == t)
                            .map(|x| x["prompt_tokens"].as_f64().unwrap_or(0.0))
                            .collect();
                        (
                            (same.iter().sum::<f64>() / same.len().max(1) as f64).round(),
                            r["ttft_p50"].as_f64().unwrap_or(0.0),
                        )
                    })
                    .collect(),
            )
        })
        .collect();
    let table: String = rows
        .iter()
        .map(|r| {
            let v = serde_json::to_value(r).unwrap();
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{} / {}</td><td>{} / {} / {}</td><td>{:.1}</td><td><b>{:.1}</b></td><td>{}</td></tr>",
                v["prompt_tokens"], v["concurrency"], v["requests"],
                ms(v["ttft_p50"].as_f64().unwrap_or(0.0)), ms(v["ttft_p95"].as_f64().unwrap_or(0.0)),
                ms(v["itl_p50"].as_f64().unwrap_or(0.0)), ms(v["itl_p95"].as_f64().unwrap_or(0.0)), ms(v["itl_p99"].as_f64().unwrap_or(0.0)),
                v["request_tps"].as_f64().unwrap_or(0.0), v["aggregate_tps"].as_f64().unwrap_or(0.0), v["errors"]
            )
        })
        .collect();
    format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tendril bench · {model}</title>
<style>
:root{{--bg:#f7f7f5;--panel:#fff;--text:#1c1c1a;--muted:#6b6b66;--line:#e3e3de}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111210;--panel:#181917;--text:#ececea;--muted:#9a9a93;--line:#2c2d2a}}}}
body{{background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;margin:0}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px 60px}}
h1{{font-size:24px;margin:0 0 4px;letter-spacing:-.02em}} h2{{font-size:14px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}}
.sub{{color:var(--muted);font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px;margin-top:20px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px}}
.chart{{width:100%;height:auto;color:var(--text)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}} th,td{{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}} th{{color:var(--muted);font-weight:600}}
.wide{{grid-column:1/-1;overflow-x:auto}}
.bd{{display:grid;grid-template-columns:170px 1fr;gap:10px;align-items:center;margin:8px 0}} .bn{{font-size:13px;text-align:right;color:var(--muted)}}
.bbars{{display:grid;grid-template-columns:1fr 70px;gap:3px 8px;align-items:center;font-size:12px}}
.bar{{height:10px;border-radius:5px}} .bar.p,.sw.p{{background:#94a3b8}} .bar.m,.sw.m{{background:#16a34a}}
.legend{{font-size:12px;color:var(--muted);margin-bottom:6px}} .sw{{display:inline-block;width:10px;height:10px;border-radius:3px;margin:0 4px 0 10px;vertical-align:-1px}}
</style></head><body><main>
<h1>Tendril bench · {model}</h1>
<div class="sub">{plan_label} · predicted {pred_tps:.1} tok/s per conversation · generated {now}</div>
<div class="grid">
{tput}
{ttft}
{breakdown}
{hist}
<section class="wide"><h2>All results</h2><table><thead><tr><th>Prompt tok</th><th>Concurrency</th><th>Requests</th><th>First token p50 / p95</th><th>Inter-token p50 / p95 / p99</th><th>Tok/s each</th><th>Tok/s total</th><th>Errors</th></tr></thead><tbody>{table}</tbody></table></section>
</div></main></body></html>"#,
        model = esc(model),
        plan_label = esc(plan["label"].as_str().unwrap_or("")),
        pred_tps = plan["tokens_per_sec"].as_f64().unwrap_or(0.0),
        now = tendril_cluster::coordinator::clock(),
        tput = line_chart(
            "Total throughput vs concurrency",
            "concurrent requests",
            "tokens / s",
            &tput,
            |v| format!("{v:.0}")
        ),
        ttft = line_chart(
            "Time to first token vs prompt length",
            "prompt tokens",
            "p50",
            &ttft,
            ms
        ),
        breakdown = breakdown(tel),
        hist = histogram("Inter-token latency", itl),
        table = table
    )
}
