//! Terminal presentation helpers: colors (respecting NO_COLOR and pipes),
//! aligned tables and memory bars.

use owo_colors::{OwoColorize, Stream};
use std::fmt::Display;

pub fn bold(s: impl Display) -> String {
    format!("{}", s.if_supports_color(Stream::Stdout, |t| t.bold()))
}
pub fn dim(s: impl Display) -> String {
    format!("{}", s.if_supports_color(Stream::Stdout, |t| t.dimmed()))
}
pub fn green(s: impl Display) -> String {
    format!("{}", s.if_supports_color(Stream::Stdout, |t| t.green()))
}
pub fn red(s: impl Display) -> String {
    format!("{}", s.if_supports_color(Stream::Stdout, |t| t.red()))
}
pub fn yellow(s: impl Display) -> String {
    format!("{}", s.if_supports_color(Stream::Stdout, |t| t.yellow()))
}
pub fn cyan(s: impl Display) -> String {
    format!("{}", s.if_supports_color(Stream::Stdout, |t| t.cyan()))
}
pub fn magenta(s: impl Display) -> String {
    format!("{}", s.if_supports_color(Stream::Stdout, |t| t.magenta()))
}

pub fn ok_mark() -> String {
    green("✓")
}
pub fn bad_mark() -> String {
    red("✗")
}
pub fn warn_mark() -> String {
    yellow("!")
}

/// Section heading.
pub fn heading(title: &str) {
    println!();
    println!("{}", bold(title));
}

/// "  Label     value" rows with aligned labels.
pub fn kv(rows: &[(&str, String)]) {
    let w = rows
        .iter()
        .map(|(k, _)| k.chars().count())
        .max()
        .unwrap_or(0);
    for (k, v) in rows {
        println!("  {}  {}", dim(format!("{k:<w$}")), v);
    }
}

/// Visible width of a string, ignoring ANSI escapes.
pub fn visible_width(s: &str) -> usize {
    let mut w = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if in_esc {
            if c == 'm' {
                in_esc = false;
            }
        } else if c == '\x1b' {
            in_esc = true;
        } else {
            w += 1;
        }
    }
    w
}

fn pad(s: &str, w: usize, right: bool) -> String {
    let n = w.saturating_sub(visible_width(s));
    if right {
        format!("{}{}", " ".repeat(n), s)
    } else {
        format!("{}{}", s, " ".repeat(n))
    }
}

pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    right: Vec<bool>,
    indent: usize,
}

impl Table {
    pub fn new(headers: &[&str]) -> Table {
        Table {
            headers: headers.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
            right: vec![false; headers.len()],
            indent: 2,
        }
    }
    pub fn right(mut self, cols: &[usize]) -> Table {
        for &c in cols {
            if c < self.right.len() {
                self.right[c] = true;
            }
        }
        self
    }
    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }
    pub fn print(&self) {
        let n = self.headers.len();
        let mut w = vec![0usize; n];
        for (i, h) in self.headers.iter().enumerate() {
            w[i] = visible_width(h);
        }
        for r in &self.rows {
            for (i, c) in r.iter().enumerate().take(n) {
                w[i] = w[i].max(visible_width(c));
            }
        }
        let ind = " ".repeat(self.indent);
        let hdr: Vec<String> = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| pad(h, w[i], self.right[i]))
            .collect();
        println!("{ind}{}", dim(hdr.join("  ").trim_end()));
        for r in &self.rows {
            let cells: Vec<String> = (0..n)
                .map(|i| {
                    pad(
                        r.get(i).map(|s| s.as_str()).unwrap_or(""),
                        w[i],
                        self.right[i],
                    )
                })
                .collect();
            println!("{ind}{}", cells.join("  ").trim_end());
        }
    }
}

/// A memory bar: filled share of `used/total`, colored by pressure.
pub fn bar(used: f64, total: f64, width: usize) -> String {
    let frac = if total > 0.0 {
        (used / total).clamp(0.0, 1.5)
    } else {
        1.5
    };
    let filled = ((frac.min(1.0)) * width as f64).round() as usize;
    let s = format!("{}{}", "█".repeat(filled), "░".repeat(width - filled));
    if frac > 1.0 {
        red(s)
    } else if frac > 0.9 {
        yellow(s)
    } else {
        green(s)
    }
}

/// Word-wrap `text` to the terminal with a hanging indent.
pub fn wrap(text: &str, indent: usize, width: usize) -> String {
    let mut out = String::new();
    let mut line = 0usize;
    let pad = " ".repeat(indent);
    for word in text.split_whitespace() {
        let wl = visible_width(word);
        if line > 0 && line + 1 + wl > width.saturating_sub(indent) {
            out.push('\n');
            out.push_str(&pad);
            line = 0;
        } else if line > 0 {
            out.push(' ');
            line += 1;
        }
        out.push_str(word);
        line += wl;
    }
    out
}

pub fn term_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|c| c.parse().ok())
        .unwrap_or(100)
        .clamp(60, 140)
}
