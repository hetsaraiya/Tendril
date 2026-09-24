//! Byte quantities, parsing and human formatting.
//!
//! Tendril reports memory in binary units (GiB) everywhere. A "16 GB" Mac has
//! 16 GiB of unified memory, and mixing decimal and binary units is the most
//! common source of "it should have fit" confusion.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Sub};

pub const KIB: u64 = 1024;
pub const MIB: u64 = 1024 * KIB;
pub const GIB: u64 = 1024 * MIB;

/// A byte count. Arithmetic saturates rather than wrapping: a planner that
/// overflows must err on the side of "does not fit".
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Bytes(pub u64);

impl Bytes {
    pub const ZERO: Bytes = Bytes(0);

    pub fn gib(v: f64) -> Bytes {
        Bytes((v * GIB as f64).max(0.0) as u64)
    }
    pub fn mib(v: f64) -> Bytes {
        Bytes((v * MIB as f64).max(0.0) as u64)
    }
    pub fn as_gib(self) -> f64 {
        self.0 as f64 / GIB as f64
    }
    pub fn as_mib(self) -> f64 {
        self.0 as f64 / MIB as f64
    }
    pub fn saturating_sub(self, o: Bytes) -> Bytes {
        Bytes(self.0.saturating_sub(o.0))
    }
    pub fn scale(self, f: f64) -> Bytes {
        Bytes((self.0 as f64 * f).max(0.0).min(u64::MAX as f64) as u64)
    }
    pub fn times(self, n: u64) -> Bytes {
        Bytes(self.0.saturating_mul(n))
    }

    /// Parse "16", "16GB", "16 GiB", "512m", "1.5t". Bare numbers are GiB.
    pub fn parse(s: &str) -> Option<Bytes> {
        let t = s.trim().to_ascii_lowercase().replace(' ', "");
        let idx = t
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(t.len());
        let (num, unit) = t.split_at(idx);
        let v: f64 = num.parse().ok()?;
        let mult = match unit.trim_end_matches('b').trim_end_matches('i') {
            "" if unit.is_empty() => GIB as f64,
            "" => 1.0,
            "k" => KIB as f64,
            "m" => MIB as f64,
            "g" => GIB as f64,
            "t" => (GIB * 1024) as f64,
            _ => return None,
        };
        if !v.is_finite() || v < 0.0 {
            return None;
        }
        Some(Bytes((v * mult) as u64))
    }
}

impl Add for Bytes {
    type Output = Bytes;
    fn add(self, o: Bytes) -> Bytes {
        Bytes(self.0.saturating_add(o.0))
    }
}
impl AddAssign for Bytes {
    fn add_assign(&mut self, o: Bytes) {
        self.0 = self.0.saturating_add(o.0);
    }
}
impl Sub for Bytes {
    type Output = Bytes;
    fn sub(self, o: Bytes) -> Bytes {
        Bytes(self.0.saturating_sub(o.0))
    }
}
impl std::iter::Sum for Bytes {
    fn sum<I: Iterator<Item = Bytes>>(iter: I) -> Bytes {
        iter.fold(Bytes::ZERO, |a, b| a + b)
    }
}

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.0 as f64;
        let (v, u) = if self.0 >= GIB {
            (b / GIB as f64, "GiB")
        } else if self.0 >= MIB {
            (b / MIB as f64, "MiB")
        } else if self.0 >= KIB {
            (b / KIB as f64, "KiB")
        } else {
            return write!(f, "{} B", self.0);
        };
        if v >= 100.0 {
            write!(f, "{v:.0} {u}")
        } else if v >= 10.0 {
            write!(f, "{v:.1} {u}")
        } else {
            write!(f, "{v:.2} {u}")
        }
    }
}

/// Format a duration given in milliseconds compactly.
pub fn fmt_ms(ms: f64) -> String {
    if !ms.is_finite() {
        "∞".into()
    } else if ms >= 60_000.0 {
        format!("{:.1} min", ms / 60_000.0)
    } else if ms >= 1000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else if ms >= 10.0 {
        format!("{ms:.0} ms")
    } else if ms >= 1.0 {
        format!("{ms:.1} ms")
    } else {
        format!("{:.0} µs", ms * 1000.0)
    }
}

/// Format a count with SI suffix (params, tokens).
pub fn fmt_count(n: u64) -> String {
    let f = n as f64;
    if f >= 1e12 {
        format!("{:.2}T", f / 1e12)
    } else if f >= 1e9 {
        format!("{:.2}B", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.1}M", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.1}K", f / 1e3)
    } else {
        n.to_string()
    }
}

/// Parse a token count like "8192", "8k", "128K", "1m".
pub fn parse_tokens(s: &str) -> Option<u64> {
    let t = s.trim().to_ascii_lowercase();
    let (num, mult) = if let Some(n) = t.strip_suffix('k') {
        (n, 1024.0)
    } else if let Some(n) = t.strip_suffix('m') {
        (n, 1024.0 * 1024.0)
    } else {
        (t.as_str(), 1.0)
    };
    let v: f64 = num.parse().ok()?;
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    Some((v * mult).round() as u64)
}

pub fn fmt_tokens(n: u64) -> String {
    if n >= 1024 && n.is_multiple_of(1024) {
        format!("{}K", n / 1024)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_units() {
        assert_eq!(Bytes::parse("16"), Some(Bytes(16 * GIB)));
        assert_eq!(Bytes::parse("16GB"), Some(Bytes(16 * GIB)));
        assert_eq!(Bytes::parse("16 GiB"), Some(Bytes(16 * GIB)));
        assert_eq!(Bytes::parse("512m"), Some(Bytes(512 * MIB)));
        assert_eq!(Bytes::parse("100b"), Some(Bytes(100)));
        assert_eq!(Bytes::parse("x"), None);
        assert_eq!(parse_tokens("8k"), Some(8192));
        assert_eq!(parse_tokens("32768"), Some(32768));
    }

    #[test]
    fn saturating() {
        assert_eq!(Bytes(u64::MAX) + Bytes(5), Bytes(u64::MAX));
        assert_eq!(Bytes(3) - Bytes(5), Bytes(0));
    }

    #[test]
    fn display() {
        assert_eq!(Bytes(16 * GIB).to_string(), "16.0 GiB");
        assert_eq!(Bytes(512 * MIB).to_string(), "512 MiB");
        assert_eq!(fmt_tokens(8192), "8K");
    }
}
