//! A cluster description for planning: nodes plus the links between them.

use crate::hardware::NodeProfile;
use crate::presets;
use crate::units::Bytes;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub name: String,
    /// Usable throughput, Gbit/s.
    pub bandwidth_gbps: f64,
    /// Round-trip time, milliseconds (includes stack overhead, not just wire).
    pub rtt_ms: f64,
    #[serde(default)]
    pub measured: bool,
}

impl Link {
    pub fn preset(kind: &str) -> Option<Link> {
        let (name, bw, rtt) = match kind
            .to_ascii_lowercase()
            .replace(['-', '_', ' '], "")
            .as_str()
        {
            "local" | "loopback" | "same" => ("same machine", 80.0, 0.02),
            "tb5" | "thunderbolt5" => ("Thunderbolt 5", 40.0, 0.06),
            "tb" | "tb4" | "thunderbolt" | "thunderbolt4" | "tb3" => ("Thunderbolt 4", 18.0, 0.08),
            "25gbe" | "25g" => ("25 GbE", 23.0, 0.05),
            "10gbe" | "10g" => ("10 GbE", 9.4, 0.1),
            "5gbe" | "5g" => ("5 GbE", 4.7, 0.15),
            "2.5gbe" | "25gbe2" | "2.5g" => ("2.5 GbE", 2.35, 0.2),
            "gbe" | "1gbe" | "ethernet" | "1g" | "lan" => ("1 GbE", 0.94, 0.25),
            "wifi7" => ("Wi-Fi 7", 1.5, 2.0),
            "wifi" | "wifi6" | "wifi6e" => ("Wi-Fi 6", 0.6, 3.0),
            "wifi5" => ("Wi-Fi 5", 0.3, 4.0),
            "infiniband" | "ib" => ("InfiniBand", 180.0, 0.01),
            _ => return None,
        };
        Some(Link {
            name: name.into(),
            bandwidth_gbps: bw,
            rtt_ms: rtt,
            measured: false,
        })
    }

    pub fn names() -> &'static str {
        "tb5, thunderbolt, 25gbe, 10gbe, 5gbe, 2.5gbe, gbe, wifi7, wifi, wifi5, infiniband"
    }

    /// One-way time to move `bytes`, ms.
    pub fn transfer_ms(&self, bytes: u64) -> f64 {
        // Half the RTT for the one-way hop, serialization at link speed, and a
        // fixed framing/syscall cost measured from typical TCP stacks.
        self.rtt_ms / 2.0 + (bytes as f64 * 8.0) / (self.bandwidth_gbps * 1e9) * 1e3 + 0.03
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cluster {
    pub nodes: Vec<NodeProfile>,
    /// Links keyed by (i, j) node indices with i < j.
    #[serde(default)]
    pub links: BTreeMap<String, Link>,
    pub default_link: Link,
}

impl Cluster {
    pub fn new(nodes: Vec<NodeProfile>, default_link: Link) -> Cluster {
        Cluster {
            nodes,
            links: BTreeMap::new(),
            default_link,
        }
    }

    fn key(a: &str, b: &str) -> String {
        if a <= b {
            format!("{a}<->{b}")
        } else {
            format!("{b}<->{a}")
        }
    }

    pub fn set_link(&mut self, a: &str, b: &str, link: Link) {
        self.links.insert(Self::key(a, b), link);
    }

    pub fn link(&self, a: usize, b: usize) -> &Link {
        let (na, nb) = (&self.nodes[a].name, &self.nodes[b].name);
        self.links
            .get(&Self::key(na, nb))
            .unwrap_or(&self.default_link)
    }

    pub fn total_usable(&self) -> Bytes {
        self.nodes.iter().map(|n| n.usable_memory).sum()
    }

    /// Make node names unique ("m4", "m4-2") so they can be referenced.
    pub fn dedupe_names(&mut self) {
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        for n in &mut self.nodes {
            let c = seen.entry(n.name.clone()).or_insert(0);
            *c += 1;
            if *c > 1 {
                n.name = format!("{}-{}", n.name, c);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster files

#[derive(Debug, Deserialize)]
struct FileNode {
    name: String,
    hardware: Option<String>,
    memory: Option<String>,
    usable_memory: Option<String>,
    bandwidth_gbs: Option<f64>,
    tflops: Option<f64>,
    address: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FileLink {
    between: [String; 2],
    #[serde(rename = "type")]
    kind: Option<String>,
    bandwidth_gbps: Option<f64>,
    rtt_ms: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ClusterFile {
    #[serde(default)]
    link: Option<String>,
    #[serde(default, rename = "node")]
    nodes: Vec<FileNode>,
    #[serde(default, rename = "links")]
    links: Vec<FileLink>,
}

/// Load a TOML cluster description. See `examples/two-macs.toml`.
pub fn load_cluster_file(path: &Path) -> Result<Cluster> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    parse_cluster_toml(&text).with_context(|| format!("in {}", path.display()))
}

pub fn parse_cluster_toml(text: &str) -> Result<Cluster> {
    let f: ClusterFile = toml::from_str(text)?;
    if f.nodes.is_empty() {
        bail!("cluster file defines no [[node]] entries");
    }
    let default_link = match &f.link {
        Some(k) => Link::preset(k)
            .with_context(|| format!("unknown link type '{k}' (try {})", Link::names()))?,
        None => Link::preset("gbe").unwrap(),
    };
    let mut nodes = Vec::new();
    for n in &f.nodes {
        let spec = match (&n.hardware, &n.memory) {
            (Some(h), Some(m)) if !h.contains(':') => format!("{h}:{m}"),
            (Some(h), _) => h.clone(),
            (None, _) => "cpu".into(),
        };
        let mut p = presets::node_from_spec(&n.name, &spec).map_err(anyhow::Error::msg)?;
        if let Some(bw) = n.bandwidth_gbs {
            p.bandwidth_gbs = bw;
            p.source = crate::hardware::ProfileSource::Manual;
        }
        if let Some(t) = n.tflops {
            p.tflops = t;
        }
        if let Some(u) = &n.usable_memory {
            p.usable_memory =
                Bytes::parse(u).with_context(|| format!("bad usable_memory '{u}'"))?;
            p.usable_reason = "set in cluster file".into();
        }
        p.address = n.address.clone();
        nodes.push(p);
    }
    let mut c = Cluster::new(nodes, default_link);
    for l in &f.links {
        let base = match &l.kind {
            Some(k) => Link::preset(k).with_context(|| format!("unknown link type '{k}'"))?,
            None => c.default_link.clone(),
        };
        let link = Link {
            name: base.name.clone(),
            bandwidth_gbps: l.bandwidth_gbps.unwrap_or(base.bandwidth_gbps),
            rtt_ms: l.rtt_ms.unwrap_or(base.rtt_ms),
            measured: false,
        };
        for end in &l.between {
            if !c.nodes.iter().any(|n| &n.name == end) {
                bail!("link references unknown node '{end}'");
            }
        }
        c.set_link(&l.between[0], &l.between[1], link);
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_file() {
        let c = parse_cluster_toml(
            r#"
            link = "wifi"
            [[node]]
            name = "air"
            hardware = "m4"
            memory = "16gb"
            [[node]]
            name = "pro"
            hardware = "m4-pro:48gb"
            [[links]]
            between = ["air", "pro"]
            type = "thunderbolt"
            "#,
        )
        .unwrap();
        assert_eq!(c.nodes.len(), 2);
        assert_eq!(c.link(0, 1).name, "Thunderbolt 4");
        assert_eq!(c.default_link.name, "Wi-Fi 6");
        assert!(parse_cluster_toml("[[node]]\nname='x'\nhardware='zzz'").is_err());
    }
}
