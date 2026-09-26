//! Placement search: which machines run which contiguous slice of layers.
//!
//! For each ordered subset of nodes we solve an exact dynamic program over
//! contiguous layer cuts (every legal boundary is considered, including
//! embedding/head placement). Candidates are then ranked by the requested
//! goal, and every rejected alternative carries a reason.

use crate::cluster::Cluster;
use crate::hardware::{Backend, NodeProfile};
use crate::model::ModelSpec;
use crate::units::{Bytes, MIB};
use serde::{Deserialize, Serialize};

/// What the user intends to run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Workload {
    /// Maximum tokens per sequence (prompt + output).
    pub context: u64,
    /// Sequences served at the same time.
    pub concurrency: u64,
    /// Typical prompt length, for time-to-first-token.
    pub prompt_tokens: u64,
    /// Typical output length, for average decode position.
    pub output_tokens: u64,
    /// Bytes per KV element (2 = f16/bf16).
    pub kv_elem_bytes: u64,
    /// Prefill is processed in chunks of this many tokens.
    pub prefill_chunk: u64,
}

impl Workload {
    pub fn new(context: u64, concurrency: u64) -> Workload {
        let context = context.max(16);
        let prompt = (context / 2).clamp(1, 1024);
        let output = (context - prompt).clamp(1, 256);
        Workload {
            context,
            concurrency: concurrency.max(1),
            prompt_tokens: prompt,
            output_tokens: output,
            kv_elem_bytes: 2,
            prefill_chunk: 512,
        }
    }
    fn decode_position(&self) -> u64 {
        (self.prompt_tokens + self.output_tokens / 2).min(self.context)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Goal {
    /// Good single-user latency with healthy memory headroom (default).
    #[default]
    Balanced,
    /// Fastest tokens/s for one conversation.
    Latency,
    /// Most tokens/s across all concurrent requests.
    Throughput,
    /// Most memory headroom on every node.
    Memory,
}

impl Goal {
    pub fn parse(s: &str) -> Option<Goal> {
        Some(match s.to_ascii_lowercase().as_str() {
            "balanced" | "default" => Goal::Balanced,
            "latency" | "speed" | "fast" | "lowest-latency" => Goal::Latency,
            "throughput" | "tps" | "highest-throughput" => Goal::Throughput,
            "memory" | "headroom" | "safe" => Goal::Memory,
            _ => return None,
        })
    }
    pub fn label(self) -> &'static str {
        match self {
            Goal::Balanced => "balanced",
            Goal::Latency => "lowest latency",
            Goal::Throughput => "highest throughput",
            Goal::Memory => "most memory headroom",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanOptions {
    pub goal: Goal,
    /// Fraction of each node's usable memory kept free as a safety margin.
    pub safety_frac: f64,
    /// Upper bound on the number of nodes in one pipeline.
    pub max_stages: usize,
    /// Cap on attention scratch; the engine chunks prefill to respect it.
    pub attn_scratch_cap: Bytes,
}

impl Default for PlanOptions {
    fn default() -> Self {
        PlanOptions {
            goal: Goal::Balanced,
            safety_frac: 0.05,
            max_stages: 8,
            attn_scratch_cap: Bytes(256 * MIB),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemoryBreakdown {
    pub weights: Bytes,
    pub kv: Bytes,
    pub scratch: Bytes,
    pub transport: Bytes,
    pub runtime: Bytes,
    /// Extra transient while loading (largest tensor being converted).
    pub load_transient: Bytes,
    pub steady: Bytes,
    pub peak: Bytes,
    pub margin: Bytes,
    pub usable: Bytes,
}

impl MemoryBreakdown {
    pub fn fits(&self) -> bool {
        self.peak + self.margin <= self.usable
    }
    /// Required minus allowed; zero when it fits.
    pub fn shortfall(&self) -> Bytes {
        (self.peak + self.margin).saturating_sub(self.usable)
    }
    pub fn headroom(&self) -> Bytes {
        self.usable.saturating_sub(self.peak)
    }
    pub fn pressure(&self) -> f64 {
        if self.usable.0 == 0 {
            f64::INFINITY
        } else {
            self.peak.0 as f64 / self.usable.0 as f64
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StagePlan {
    pub node: usize,
    pub node_name: String,
    pub backend: Backend,
    /// Transformer layers [start, end).
    pub layer_start: u64,
    pub layer_end: u64,
    pub embed: bool,
    pub head: bool,
    pub mem: MemoryBreakdown,
    /// Per-token decode compute, single sequence, ms.
    pub decode_ms: f64,
    /// Decode step at the planned concurrency, ms.
    pub decode_ms_batch: f64,
    /// Prefill of one typical prompt, ms.
    pub prefill_ms: f64,
}

impl StagePlan {
    pub fn layers(&self) -> u64 {
        self.layer_end - self.layer_start
    }
    pub fn describe_components(&self) -> String {
        let mut parts = Vec::new();
        if self.embed {
            parts.push("embeddings".to_string());
        }
        if self.layers() > 0 {
            parts.push(if self.layers() == 1 {
                format!("layer {}", self.layer_start)
            } else {
                format!("layers {}–{}", self.layer_start, self.layer_end - 1)
            });
        }
        if self.head {
            parts.push("final norm + LM head".to_string());
        }
        parts.join(", ")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub stages: Vec<StagePlan>,
    /// Single-sequence time per output token, ms.
    pub decode_ms: f64,
    /// Portion of `decode_ms` spent on the network.
    pub network_ms: f64,
    /// Time to first token for a typical prompt, ms.
    pub ttft_ms: f64,
    pub tokens_per_sec: f64,
    /// Aggregate tokens/s at the planned concurrency.
    pub throughput_tps: f64,
    /// Stage index with the largest decode time.
    pub bottleneck: usize,
    pub boundary_bytes_per_token: u64,
    pub max_pressure: f64,
    pub score: f64,
    pub score_parts: Vec<(String, f64)>,
}

impl Plan {
    pub fn node_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.node_name.as_str()).collect()
    }
    pub fn label(&self) -> String {
        if self.stages.len() == 1 {
            format!("{} alone", self.stages[0].node_name)
        } else {
            self.node_names().join(" → ")
        }
    }
}

/// A candidate node ordering that could not host the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rejection {
    pub nodes: Vec<String>,
    pub reason: String,
    /// Smallest possible worst-node shortfall over all cuts.
    pub shortfall: Bytes,
}

/// One row of the two-node cut table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CutRow {
    /// Layers [0, cut) on the first node.
    pub cut: u64,
    pub first: MemoryBreakdown,
    pub second: MemoryBreakdown,
    pub decode_ms: Option<f64>,
    pub selected: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Exclusion {
    pub node: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanResult {
    pub workload: Workload,
    pub goal: Goal,
    pub selected: Option<Plan>,
    /// Other good feasible plans, best first (excluding the selection).
    pub alternatives: Vec<Plan>,
    pub rejected: Vec<Rejection>,
    pub excluded: Vec<Exclusion>,
    pub cut_table: Vec<CutRow>,
    pub cut_table_nodes: Vec<String>,
    pub orders_evaluated: usize,
    pub cuts_evaluated: u64,
}

// ---------------------------------------------------------------------------

/// Precomputed per-model/per-workload prefix sums.
struct Prep<'a> {
    m: &'a ModelSpec,
    w: &'a Workload,
    o: &'a PlanOptions,
    /// prefix sums, length L+1
    layer_bytes: Vec<u64>,
    layer_params: Vec<u64>,
    kv_ctx: Vec<u64>,
    kv_read: Vec<u64>,
    head_params: u64,
    scratch_layers: Bytes,
}

impl<'a> Prep<'a> {
    fn new(m: &'a ModelSpec, w: &'a Workload, o: &'a PlanOptions) -> Prep<'a> {
        let l = m.num_layers as usize;
        let mut layer_bytes = vec![0u64; l + 1];
        let mut layer_params = vec![0u64; l + 1];
        let mut kv_ctx = vec![0u64; l + 1];
        let mut kv_read = vec![0u64; l + 1];
        let pos = w.decode_position();
        for i in 0..l {
            layer_bytes[i + 1] = layer_bytes[i] + m.bytes.layers.get(i).map_or(0, |b| b.0);
            layer_params[i + 1] = layer_params[i] + m.params.layers.get(i).copied().unwrap_or(0);
            kv_ctx[i + 1] = kv_ctx[i]
                + m.kv_bytes(i..i + 1, w.context, w.concurrency, w.kv_elem_bytes)
                    .0;
            kv_read[i + 1] = kv_read[i] + m.kv_bytes(i..i + 1, pos, 1, w.kv_elem_bytes).0;
        }
        let chunk = w.prefill_chunk.min(w.context);
        let attn = Bytes(m.num_heads * chunk * w.context * 4).min(o.attn_scratch_cap);
        let mlp = Bytes(chunk * m.intermediate_size * 4 * 3);
        let resid = Bytes(chunk * m.hidden_size * 4 * 6);
        Prep {
            m,
            w,
            o,
            layer_bytes,
            layer_params,
            kv_ctx,
            kv_read,
            head_params: if m.tie_embeddings {
                m.params.embed
            } else {
                m.params.lm_head
            },
            scratch_layers: attn + mlp + resid,
        }
    }

    fn memory(
        &self,
        node: &NodeProfile,
        a: usize,
        b: usize,
        first: bool,
        last: bool,
    ) -> MemoryBreakdown {
        let m = self.m;
        let mut weights = Bytes(self.layer_bytes[b] - self.layer_bytes[a]);
        if first {
            weights += m.bytes.embed;
        }
        if last {
            weights += m.bytes.final_norm;
            if !(first && m.tie_embeddings) {
                weights += m.head_bytes();
            }
        }
        let kv = Bytes(self.kv_ctx[b] - self.kv_ctx[a]);
        let mut scratch = if b > a {
            self.scratch_layers
        } else {
            Bytes(16 * MIB)
        };
        if last {
            scratch += Bytes(m.vocab_size * 4 * self.w.concurrency.max(1) * 2);
        }
        let transport = if first && last {
            Bytes::ZERO
        } else {
            Bytes(2 * self.w.prefill_chunk * m.hidden_size * 2 + 8 * MIB)
        };
        let runtime = node.backend.runtime_overhead();
        let load_transient = m.bytes.largest_tensor.min(weights);
        let steady = weights + kv + scratch + transport + runtime;
        let peak = steady.max(weights + runtime + load_transient);
        let margin = node.usable_memory.scale(self.o.safety_frac);
        MemoryBreakdown {
            weights,
            kv,
            scratch,
            transport,
            runtime,
            load_transient,
            steady,
            peak,
            margin,
            usable: node.usable_memory,
        }
    }

    /// Decode compute for one step with `batch` sequences, ms.
    fn decode_ms(&self, node: &NodeProfile, a: usize, b: usize, last: bool, batch: u64) -> f64 {
        let m = self.m;
        let mut read = (self.layer_bytes[b] - self.layer_bytes[a]) as f64;
        let mut params = (self.layer_params[b] - self.layer_params[a]) as f64;
        if last {
            read += (m.head_bytes() + m.bytes.final_norm).0 as f64;
            params += self.head_params as f64;
        }
        read += (self.kv_read[b] - self.kv_read[a]) as f64 * batch as f64;
        let mem_ms = read / (node.effective_bandwidth_gbs() * 1e9) * 1e3;
        let flops = 2.0 * params * batch as f64;
        let compute_ms = flops / (node.effective_tflops() * 1e12) * 1e3;
        let mut ms = mem_ms.max(compute_ms) + (b - a) as f64 * node.backend.per_layer_overhead_ms();
        if last {
            ms += 0.08; // sampling
        }
        ms
    }

    /// Prefill of `tokens` tokens, ms.
    fn prefill_ms(&self, node: &NodeProfile, a: usize, b: usize, last: bool, tokens: u64) -> f64 {
        let m = self.m;
        let mut params = (self.layer_params[b] - self.layer_params[a]) as f64;
        let weight_bytes = (self.layer_bytes[b] - self.layer_bytes[a]) as f64;
        let t = tokens as f64;
        // Causal attention: ~2 * 2 * t^2/2 * q_dim per layer.
        let attn = 2.0 * t * t * (m.num_heads * m.head_dim) as f64 * (b - a) as f64;
        if last {
            params += self.head_params as f64 / t.max(1.0); // head only on the last token
        }
        let flops = 2.0 * params * t + attn;
        let compute_ms = flops / (node.effective_tflops() * 1e12) * 1e3;
        let chunks = tokens.div_ceil(self.w.prefill_chunk.max(1)) as f64;
        let mem_ms = weight_bytes * chunks / (node.effective_bandwidth_gbs() * 1e9) * 1e3;
        compute_ms.max(mem_ms) + chunks * (b - a) as f64 * node.backend.per_layer_overhead_ms()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum DpMode {
    /// Minimize the sum of stage decode times (with a pressure penalty).
    Latency { batch: u64, pressure_penalty: bool },
    /// Minimize the worst stage memory pressure.
    Pressure,
    /// Minimize the worst stage shortfall (for explaining infeasibility).
    Shortfall,
}

/// Solve the best contiguous partition of all layers onto `order`.
/// Returns stage ranges.
fn dp(prep: &Prep, nodes: &[&NodeProfile], mode: DpMode) -> Option<(f64, Vec<(usize, usize)>)> {
    let l = prep.m.num_layers as usize;
    let s = nodes.len();
    if s == 0 || s > l.max(1) {
        return None;
    }
    let inf = f64::INFINITY;
    // best[k][j]: cost of placing layers [0, j) on the first k nodes.
    let mut best = vec![vec![inf; l + 1]; s + 1];
    let mut from = vec![vec![usize::MAX; l + 1]; s + 1];
    best[0][0] = 0.0;
    let combine = |acc: f64, c: f64| match mode {
        DpMode::Latency { .. } => acc + c,
        DpMode::Pressure | DpMode::Shortfall => acc.max(c),
    };
    for k in 1..=s {
        let node = nodes[k - 1];
        let first = k == 1;
        let last = k == s;
        let j_range = if last {
            l..=l
        } else {
            k..=l.saturating_sub(s - k)
        };
        for j in j_range {
            let i_lo = k - 1;
            for i in i_lo..j {
                let acc = best[k - 1][i];
                if acc == inf {
                    continue;
                }
                let mem = prep.memory(node, i, j, first, last);
                let c = match mode {
                    DpMode::Latency {
                        batch,
                        pressure_penalty,
                    } => {
                        if !mem.fits() {
                            continue;
                        }
                        let mut t = prep.decode_ms(node, i, j, last, batch);
                        if pressure_penalty {
                            let p = mem.pressure();
                            if p > 0.85 {
                                t *= 1.0 + (p - 0.85) * 1.5;
                            }
                        }
                        t
                    }
                    DpMode::Pressure => {
                        if !mem.fits() {
                            continue;
                        }
                        mem.pressure()
                    }
                    DpMode::Shortfall => mem.shortfall().0 as f64,
                };
                let v = combine(acc, c);
                if v < best[k][j] {
                    best[k][j] = v;
                    from[k][j] = i;
                }
            }
        }
    }
    if best[s][l] == inf {
        return None;
    }
    let mut ranges = Vec::with_capacity(s);
    let mut j = l;
    for k in (1..=s).rev() {
        let i = from[k][j];
        ranges.push((i, j));
        j = i;
    }
    ranges.reverse();
    Some((best[s][l], ranges))
}

fn build_plan(prep: &Prep, cluster: &Cluster, order: &[usize], ranges: &[(usize, usize)]) -> Plan {
    let w = prep.w;
    let m = prep.m;
    let s = order.len();
    let mut stages = Vec::with_capacity(s);
    for (k, (&ni, &(a, b))) in order.iter().zip(ranges).enumerate() {
        let node = &cluster.nodes[ni];
        let first = k == 0;
        let last = k == s - 1;
        stages.push(StagePlan {
            node: ni,
            node_name: node.name.clone(),
            backend: node.backend,
            layer_start: a as u64,
            layer_end: b as u64,
            embed: first,
            head: last,
            mem: prep.memory(node, a, b, first, last),
            decode_ms: prep.decode_ms(node, a, b, last, 1),
            decode_ms_batch: prep.decode_ms(node, a, b, last, w.concurrency),
            prefill_ms: prep.prefill_ms(node, a, b, last, w.prompt_tokens),
        });
    }
    let bpt = m.boundary_bytes_per_token() + 64;
    let mut network_1 = 0.0;
    let mut network_c = 0.0;
    for k in 0..s.saturating_sub(1) {
        let link = cluster.link(order[k], order[k + 1]);
        network_1 += link.transfer_ms(bpt);
        network_c += link.transfer_ms(bpt * w.concurrency);
    }
    if s > 1 {
        // Sampled token returns from the last stage to the first.
        let link = cluster.link(order[s - 1], order[0]);
        network_1 += link.transfer_ms(64);
        network_c += link.transfer_ms(64 * w.concurrency);
    }
    let compute_1: f64 = stages.iter().map(|st| st.decode_ms).sum();
    let compute_c: f64 = stages.iter().map(|st| st.decode_ms_batch).sum();
    let decode_ms = compute_1 + network_1;
    let decode_c = compute_c + network_c;

    // Chunked prefill pipelines chunks through the stages.
    let chunk = w.prefill_chunk.min(w.prompt_tokens).max(1);
    let chunks = w.prompt_tokens.div_ceil(chunk);
    let mut per_chunk = Vec::with_capacity(s);
    let mut fill = 0.0;
    for k in 0..s {
        let node = &cluster.nodes[order[k]];
        let (a, b) = ranges[k];
        let t = prep.prefill_ms(node, a, b, k == s - 1, chunk);
        let xfer = if k + 1 < s {
            cluster
                .link(order[k], order[k + 1])
                .transfer_ms(chunk * m.hidden_size * 2)
        } else {
            0.0
        };
        fill += t + xfer;
        per_chunk.push(t.max(xfer));
    }
    let slowest = per_chunk.iter().cloned().fold(0.0, f64::max);
    let ret = if s > 1 {
        cluster.link(order[s - 1], order[0]).transfer_ms(64)
    } else {
        0.0
    };
    let ttft = fill + (chunks.saturating_sub(1)) as f64 * slowest + ret;

    let bottleneck = stages
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.decode_ms.total_cmp(&b.1.decode_ms))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let max_pressure = stages
        .iter()
        .map(|st| st.mem.pressure())
        .fold(0.0, f64::max);
    Plan {
        stages,
        decode_ms,
        network_ms: network_1,
        ttft_ms: ttft,
        tokens_per_sec: 1000.0 / decode_ms,
        throughput_tps: w.concurrency as f64 * 1000.0 / decode_c,
        bottleneck,
        boundary_bytes_per_token: if s > 1 { bpt } else { 0 },
        max_pressure,
        score: 0.0,
        score_parts: Vec::new(),
    }
}

/// Node orderings to evaluate: every ordered subset for small clusters,
/// a bandwidth-sorted heuristic set for larger ones. Orderings that are
/// indistinguishable (identical hardware and links) are evaluated once.
fn orderings(cluster: &Cluster, max_stages: usize) -> Vec<Vec<usize>> {
    let n = cluster.nodes.len();
    let sig = |i: usize| {
        let nd = &cluster.nodes[i];
        format!(
            "{}|{}|{:.1}|{:.2}",
            nd.chip,
            nd.usable_memory.0,
            nd.effective_bandwidth_gbs(),
            nd.tflops
        )
    };
    let links_uniform = cluster.links.is_empty();
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut push = |o: Vec<usize>, out: &mut Vec<Vec<usize>>| {
        let key = if links_uniform {
            o.iter().map(|&i| sig(i)).collect::<Vec<_>>().join("/")
        } else {
            o.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join("/")
        };
        if seen.insert(key) {
            out.push(o);
        }
    };
    if n <= 6 {
        fn rec(
            n: usize,
            max: usize,
            cur: &mut Vec<usize>,
            used: &mut Vec<bool>,
            all: &mut Vec<Vec<usize>>,
        ) {
            if !cur.is_empty() {
                all.push(cur.clone());
            }
            if cur.len() == max {
                return;
            }
            for i in 0..n {
                if !used[i] {
                    used[i] = true;
                    cur.push(i);
                    rec(n, max, cur, used, all);
                    cur.pop();
                    used[i] = false;
                }
            }
        }
        let mut all = Vec::new();
        rec(
            n,
            max_stages.min(n),
            &mut Vec::new(),
            &mut vec![false; n],
            &mut all,
        );
        for o in all {
            push(o, &mut out);
        }
    } else {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            cluster.nodes[b]
                .effective_bandwidth_gbs()
                .total_cmp(&cluster.nodes[a].effective_bandwidth_gbs())
        });
        for i in 0..n {
            push(vec![i], &mut out);
        }
        for k in 2..=max_stages.min(n) {
            push(idx[..k].to_vec(), &mut out);
            let mut rev = idx[..k].to_vec();
            rev.reverse();
            push(rev, &mut out);
            // Largest-memory-first variant.
            let mut bym = idx.clone();
            bym.sort_by(|&a, &b| {
                cluster.nodes[b]
                    .usable_memory
                    .cmp(&cluster.nodes[a].usable_memory)
            });
            push(bym[..k].to_vec(), &mut out);
        }
    }
    out
}

/// Plan `model` onto `cluster` for `workload`.
pub fn plan(
    model: &ModelSpec,
    cluster: &Cluster,
    workload: &Workload,
    opts: &PlanOptions,
) -> PlanResult {
    let prep = Prep::new(model, workload, opts);
    let orders = orderings(cluster, opts.max_stages);
    let l = model.num_layers;
    let mut feasible: Vec<Plan> = Vec::new();
    let mut rejected: Vec<Rejection> = Vec::new();
    let mut cuts_evaluated: u64 = 0;
    let mode = match opts.goal {
        Goal::Latency => DpMode::Latency {
            batch: 1,
            pressure_penalty: false,
        },
        Goal::Balanced => DpMode::Latency {
            batch: 1,
            pressure_penalty: true,
        },
        Goal::Throughput => DpMode::Latency {
            batch: workload.concurrency,
            pressure_penalty: false,
        },
        Goal::Memory => DpMode::Pressure,
    };
    for order in &orders {
        let s = order.len() as u64;
        // Number of ways to cut L layers into s non-empty contiguous stages.
        cuts_evaluated += binom(l.saturating_sub(1), s.saturating_sub(1)).min(1 << 40);
        let nodes: Vec<&NodeProfile> = order.iter().map(|&i| &cluster.nodes[i]).collect();
        if s > l {
            continue;
        }
        match dp(&prep, &nodes, mode) {
            Some((_, ranges)) => feasible.push(build_plan(&prep, cluster, order, &ranges)),
            None => {
                let names: Vec<String> = nodes.iter().map(|n| n.name.clone()).collect();
                let (reason, shortfall) = explain_infeasible(&prep, cluster, order, &nodes);
                rejected.push(Rejection {
                    nodes: names,
                    reason,
                    shortfall,
                });
            }
        }
    }

    rank(&mut feasible, opts.goal);
    let selected = feasible.first().cloned();
    let alternatives: Vec<Plan> = feasible.iter().skip(1).take(6).cloned().collect();

    // Why was each unused node left out?
    let mut excluded = Vec::new();
    if let Some(sel) = &selected {
        for (i, n) in cluster.nodes.iter().enumerate() {
            if sel.stages.iter().any(|s| s.node == i) {
                continue;
            }
            let with: Option<&Plan> = feasible
                .iter()
                .find(|p| p.stages.iter().any(|s| s.node == i));
            let reason = match with {
                Some(p) => {
                    let delta = p.decode_ms - sel.decode_ms;
                    if delta > 0.05 {
                        format!(
                            "including it adds {} per token ({} vs {}) — its speed or link costs more than its memory is worth",
                            crate::units::fmt_ms(delta),
                            crate::units::fmt_ms(p.decode_ms),
                            crate::units::fmt_ms(sel.decode_ms)
                        )
                    } else {
                        "not needed: the model already fits on the selected nodes with enough headroom".to_string()
                    }
                }
                None => format!("no feasible plan that uses it ({} usable)", n.usable_memory),
            };
            excluded.push(Exclusion {
                node: n.name.clone(),
                reason,
            });
        }
    }

    // Two-node cut table for the selected (or best-attempted) pair.
    let mut cut_table = Vec::new();
    let mut cut_table_nodes = Vec::new();
    let pair: Option<Vec<usize>> = selected
        .as_ref()
        .filter(|p| p.stages.len() == 2)
        .map(|p| p.stages.iter().map(|s| s.node).collect())
        .or_else(|| orders.iter().find(|o| o.len() == 2).cloned());
    if let Some(pair) = pair {
        let (a, b) = (&cluster.nodes[pair[0]], &cluster.nodes[pair[1]]);
        cut_table_nodes = vec![a.name.clone(), b.name.clone()];
        let sel_cut = selected
            .as_ref()
            .filter(|p| p.stages.len() == 2 && p.stages[0].node == pair[0])
            .map(|p| p.stages[0].layer_end);
        let link = cluster.link(pair[0], pair[1]);
        let bpt = model.boundary_bytes_per_token() + 64;
        for cut in 1..model.num_layers as usize {
            let first = prep.memory(a, 0, cut, true, false);
            let second = prep.memory(b, cut, model.num_layers as usize, false, true);
            let decode = (first.fits() && second.fits()).then(|| {
                prep.decode_ms(a, 0, cut, false, 1)
                    + prep.decode_ms(b, cut, model.num_layers as usize, true, 1)
                    + link.transfer_ms(bpt)
                    + link.transfer_ms(64)
            });
            cut_table.push(CutRow {
                cut: cut as u64,
                first,
                second,
                decode_ms: decode,
                selected: sel_cut == Some(cut as u64),
            });
        }
    }

    rejected.sort_by(|a, b| {
        a.nodes
            .len()
            .cmp(&b.nodes.len())
            .then(a.shortfall.cmp(&b.shortfall))
    });
    PlanResult {
        workload: workload.clone(),
        goal: opts.goal,
        selected,
        alternatives,
        rejected,
        excluded,
        cut_table,
        cut_table_nodes,
        orders_evaluated: orders.len(),
        cuts_evaluated,
    }
}

fn binom(n: u64, k: u64) -> u64 {
    let k = k.min(n.saturating_sub(k));
    let mut r: u64 = 1;
    for i in 0..k {
        r = r.saturating_mul(n - i) / (i + 1);
    }
    r
}

fn explain_infeasible(
    prep: &Prep,
    cluster: &Cluster,
    order: &[usize],
    nodes: &[&NodeProfile],
) -> (String, Bytes) {
    let m = prep.m;
    if nodes.len() == 1 {
        let mem = prep.memory(nodes[0], 0, m.num_layers as usize, true, true);
        return (
            format!(
                "needs {} (weights {} + KV {} + buffers {}) but {} allows {}",
                mem.peak + mem.margin,
                mem.weights,
                mem.kv,
                mem.scratch + mem.transport + mem.runtime,
                nodes[0].name,
                mem.usable
            ),
            mem.shortfall(),
        );
    }
    match dp(prep, nodes, DpMode::Shortfall) {
        Some((short, ranges)) => {
            // Identify the node that is short in the least-bad split.
            let mut worst = String::new();
            let mut worst_b = Bytes::ZERO;
            for (k, &(a, b)) in ranges.iter().enumerate() {
                let mem = prep.memory(nodes[k], a, b, k == 0, k == nodes.len() - 1);
                if mem.shortfall() >= worst_b {
                    worst_b = mem.shortfall();
                    worst = nodes[k].name.clone();
                }
            }
            let total_need: Bytes = ranges
                .iter()
                .enumerate()
                .map(|(k, &(a, b))| {
                    let mem = prep.memory(nodes[k], a, b, k == 0, k == nodes.len() - 1);
                    mem.peak + mem.margin
                })
                .sum();
            let total_have: Bytes = order.iter().map(|&i| cluster.nodes[i].usable_memory).sum();
            (
                format!(
                    "no split fits: even the best cut leaves {worst} short by {} (needs {} total across these nodes, {} usable)",
                    Bytes(short as u64),
                    total_need,
                    total_have
                ),
                Bytes(short as u64),
            )
        }
        None => ("more stages than layers".into(), Bytes(u64::MAX)),
    }
}

fn rank(plans: &mut [Plan], goal: Goal) {
    if plans.is_empty() {
        return;
    }
    let best_decode = plans
        .iter()
        .map(|p| p.decode_ms)
        .fold(f64::INFINITY, f64::min);
    let best_ttft = plans
        .iter()
        .map(|p| p.ttft_ms)
        .fold(f64::INFINITY, f64::min);
    let best_tps = plans.iter().map(|p| p.throughput_tps).fold(0.0, f64::max);
    for p in plans.iter_mut() {
        let lat = p.decode_ms / best_decode;
        let ttft = p.ttft_ms / best_ttft;
        let pressure = (p.max_pressure - 0.8).max(0.0) * 5.0;
        let nodes = (p.stages.len() - 1) as f64 * 0.04;
        let tps = best_tps / p.throughput_tps.max(1e-9);
        p.score_parts = match goal {
            Goal::Balanced => vec![
                ("decode latency".into(), 0.6 * lat),
                ("time to first token".into(), 0.2 * ttft),
                ("memory pressure".into(), 0.15 * pressure),
                ("extra machines".into(), nodes),
            ],
            Goal::Latency => vec![
                ("decode latency".into(), lat),
                ("time to first token".into(), 0.01 * ttft),
            ],
            Goal::Throughput => vec![
                ("throughput".into(), tps),
                ("decode latency".into(), 0.01 * lat),
            ],
            Goal::Memory => vec![
                ("memory pressure".into(), p.max_pressure),
                ("decode latency".into(), 0.001 * lat),
            ],
        };
        p.score = p.score_parts.iter().map(|(_, v)| v).sum();
    }
    plans.sort_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then(a.stages.len().cmp(&b.stages.len()))
    });
}

/// Is there any feasible plan at all? (Fast path for advisors.)
pub fn feasible(
    model: &ModelSpec,
    cluster: &Cluster,
    workload: &Workload,
    opts: &PlanOptions,
) -> bool {
    let prep = Prep::new(model, workload, opts);
    orderings(cluster, opts.max_stages).iter().any(|order| {
        let nodes: Vec<&NodeProfile> = order.iter().map(|&i| &cluster.nodes[i]).collect();
        order.len() as u64 <= model.num_layers && dp(&prep, &nodes, DpMode::Pressure).is_some()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::Link;
    use crate::model::catalog;
    use crate::presets::node_from_spec;

    fn two_macs(link: &str) -> Cluster {
        Cluster::new(
            vec![
                node_from_spec("m4-air", "m4:16").unwrap(),
                node_from_spec("m5-air", "m5:16").unwrap(),
            ],
            Link::preset(link).unwrap(),
        )
    }

    #[test]
    fn gemma_19gb_on_two_macs() {
        // The motivating example: ~18.5 GB of weights, two 16 GB Macs.
        let m = catalog::lookup("gemma-2-9b").unwrap().spec();
        let mut c = two_macs("thunderbolt");
        let w = Workload::new(4096, 1);
        let r = plan(&m, &c, &w, &PlanOptions::default());
        // Neither Mac fits it alone with default GPU limits...
        assert!(r.rejected.iter().any(|x| x.nodes.len() == 1));
        // ...and with the default ~10.7 GiB GPU limit, two Macs aren't enough either.
        assert!(
            r.selected.is_none(),
            "default wired limit should be too small"
        );
        // Raising the GPU wired limit to 13 GiB makes the pair work.
        for n in &mut c.nodes {
            n.wired_limit = Some(Bytes::gib(13.0));
            n.recompute_usable();
        }
        let r = plan(&m, &c, &w, &PlanOptions::default());
        let sel = r.selected.expect("should fit on both with a raised limit");
        assert_eq!(sel.stages.len(), 2);
        assert!(sel.stages[0].embed && sel.stages[1].head);
        assert_eq!(sel.stages[0].layer_end, sel.stages[1].layer_start);
        assert_eq!(sel.stages[1].layer_end, m.num_layers);
        for s in &sel.stages {
            assert!(s.mem.fits());
        }
        // The faster M5 should take at least as many layers as... not required,
        // but the split must not default to 50/50 blindly: check the table exists.
        assert_eq!(r.cut_table.len() as u64, m.num_layers - 1);
        assert!(r.cut_table.iter().any(|c| c.selected));
    }

    #[test]
    fn prefers_single_node_when_it_fits() {
        let m = catalog::lookup("llama-3.2-3b").unwrap().spec();
        let c = two_macs("wifi");
        let r = plan(&m, &c, &Workload::new(4096, 1), &PlanOptions::default());
        let sel = r.selected.unwrap();
        assert_eq!(
            sel.stages.len(),
            1,
            "a 6 GiB model should not be split over Wi-Fi"
        );
        // The faster M5 wins.
        assert_eq!(sel.stages[0].node_name, "m5-air");
        assert_eq!(r.excluded.len(), 1);
    }

    #[test]
    fn context_changes_feasibility() {
        let m = catalog::lookup("llama-3.1-8b").unwrap().spec();
        let c = Cluster::new(
            vec![node_from_spec("g", "rtx4090").unwrap()],
            Link::preset("gbe").unwrap(),
        );
        let o = PlanOptions::default();
        assert!(feasible(&m, &c, &Workload::new(8192, 1), &o));
        assert!(!feasible(&m, &c, &Workload::new(131072, 4), &o));
    }

    #[test]
    fn slow_node_left_out() {
        let m = catalog::lookup("qwen2.5-7b").unwrap().spec();
        let c = Cluster::new(
            vec![
                node_from_spec("gpu", "rtx4090").unwrap(),
                node_from_spec("old", "cpu:32").unwrap(),
            ],
            Link::preset("gbe").unwrap(),
        );
        let r = plan(&m, &c, &Workload::new(8192, 1), &PlanOptions::default());
        let sel = r.selected.unwrap();
        assert_eq!(sel.node_names(), vec!["gpu"]);
        assert!(r.excluded[0].reason.contains("adds"));
    }

    #[test]
    fn many_nodes_is_fast() {
        let m = catalog::lookup("llama-3.1-70b").unwrap().spec();
        let nodes = (0..6)
            .map(|i| {
                node_from_spec(
                    &format!("n{i}"),
                    if i % 2 == 0 { "m4-pro:24" } else { "m4:16" },
                )
                .unwrap()
            })
            .collect();
        let c = Cluster::new(nodes, Link::preset("10gbe").unwrap());
        let t = std::time::Instant::now();
        let r = plan(&m, &c, &Workload::new(8192, 1), &PlanOptions::default());
        assert!(t.elapsed().as_secs_f64() < 5.0);
        assert!(r.selected.is_none() || r.selected.unwrap().stages.len() >= 2);
    }
}
