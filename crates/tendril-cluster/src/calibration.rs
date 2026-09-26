//! Remembering how fast machines really are.
//!
//! The planner starts from spec-sheet numbers. Once a pipeline runs, the
//! coordinator measures each stage's compute time per token; the ratio to the
//! prediction is stored per machine and applied to the next plan.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use tendril_core::NodeProfile;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Entry {
    /// measured / predicted time for this machine's work.
    pub factor: f64,
    pub samples: u64,
    pub updated: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Calibration {
    pub machines: BTreeMap<String, Entry>,
    #[serde(skip)]
    path: Option<PathBuf>,
}

fn key(name: &str, chip: &str) -> String {
    format!("{name} · {chip}")
}

impl Calibration {
    pub fn path() -> PathBuf {
        std::env::var("TENDRIL_CALIBRATION")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::config_dir()
                    .unwrap_or_else(std::env::temp_dir)
                    .join("tendril")
                    .join("calibration.json")
            })
    }

    pub fn load() -> Calibration {
        let path = Self::path();
        let mut c: Calibration = std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        c.path = Some(path);
        c
    }

    pub fn save(&self) {
        if let Some(p) = &self.path {
            if let Some(d) = p.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            if let Ok(j) = serde_json::to_vec_pretty(self) {
                let _ = std::fs::write(p, j);
            }
        }
    }

    /// Record an observation. The ratio is relative to the *currently
    /// calibrated* profile, so factors compose. Returns the new factor when it
    /// changed meaningfully.
    pub fn observe(&mut self, name: &str, chip: &str, ratio: f64, samples: u64) -> Option<f64> {
        if !ratio.is_finite() || ratio <= 0.0 {
            return None;
        }
        let e = self.machines.entry(key(name, chip)).or_insert(Entry {
            factor: 1.0,
            samples: 0,
            updated: String::new(),
        });
        let old = e.factor;
        // Damp: move halfway toward the measurement, clamp to a sane range.
        let new = (old * ratio.powf(0.5)).clamp(0.1, 20.0);
        e.samples += samples;
        e.updated = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();
        e.factor = new;
        // Only report meaningful changes.
        ((new / old - 1.0).abs() > 0.1).then_some(new)
    }

    /// Scale a profile's speed by what was measured on this machine.
    pub fn apply(&self, p: &mut NodeProfile) {
        if let Some(e) = self.machines.get(&key(&p.name, &p.chip)) {
            if (e.factor - 1.0).abs() > 0.05 {
                p.bandwidth_gbs /= e.factor;
                p.tflops /= e.factor;
                if let Some(m) = p.measured_bandwidth_gbs.as_mut() {
                    *m /= e.factor;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_and_applies() {
        let mut c = Calibration::default();
        let mut p = tendril_core::presets::node_from_spec("box", "m4:16").unwrap();
        let bw = p.bandwidth_gbs;
        // Measured 2x slower than predicted, repeatedly (each time relative to the calibrated profile).
        let mut ratio = 2.0;
        for _ in 0..6 {
            let f = c
                .observe("box", &p.chip, ratio, 64)
                .unwrap_or(c.machines.values().next().unwrap().factor);
            ratio = 2.0 / f;
        }
        let f = c.machines.values().next().unwrap().factor;
        assert!((f - 2.0).abs() < 0.2, "{f}");
        c.apply(&mut p);
        assert!((p.bandwidth_gbs - bw / f).abs() < 1e-9);
    }
}
