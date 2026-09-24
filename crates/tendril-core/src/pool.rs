//! Sharing one pool of machines between several models.
//!
//! Each model gets a slice of each machine's memory (a *budget*). Models are
//! placed one after another with the ordinary planner on whatever memory the
//! previous ones left; every placement order is tried (for up to five models)
//! and the best outcome wins: as many models placed as possible, the ones
//! listed first preferred, then the fastest combination.

use crate::cluster::Cluster;
use crate::model::ModelSpec;
use crate::planner::{plan, Plan, PlanOptions, Workload};
use crate::units::{Bytes, MIB};
use serde::Serialize;

/// Memory kept on top of a stage's planned need so re-planning inside the
/// budget reproduces the same placement despite rounding.
const SLACK: u64 = 32 * MIB;
/// A machine with less free memory than this is not offered to more models.
const MIN_FREE: u64 = 128 * MIB;

pub struct PoolModel<'a> {
    pub spec: &'a ModelSpec,
    pub workload: Workload,
    pub opts: PlanOptions,
    /// Machines (cluster indices) the model runs on now. It stays there
    /// unless moving is clearly faster, so a joining machine doesn't reload
    /// every model for a marginal gain.
    pub current: Vec<usize>,
}

/// How much faster a new placement must be to move a running model.
pub const MOVE_GAIN: f64 = 1.25;

#[derive(Clone, Debug, Serialize)]
pub struct Placement {
    /// The plan found inside this model's share (node indices refer to the
    /// full cluster). None if the model could not be placed.
    pub plan: Option<Plan>,
    /// Memory this model may use on each machine (cluster order); zero means
    /// the machine is not used for it.
    pub budgets: Vec<Bytes>,
    /// Why the model is not placed.
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Allocation {
    /// One entry per model, in the order they were given.
    pub placements: Vec<Placement>,
    /// Memory left unassigned on each machine.
    pub free: Vec<Bytes>,
}

impl Allocation {
    pub fn placed(&self) -> usize {
        self.placements.iter().filter(|p| p.plan.is_some()).count()
    }
}

/// Memory a stage needs so that `peak + safety margin` fits.
fn need(peak: Bytes, safety: f64) -> Bytes {
    let s = safety.clamp(0.0, 0.9);
    Bytes((peak.0 as f64 / (1.0 - s)).ceil() as u64 + SLACK)
}

fn place_in_order(models: &[PoolModel], cluster: &Cluster, order: &[usize]) -> Allocation {
    let n = cluster.nodes.len();
    let mut free: Vec<Bytes> = cluster.nodes.iter().map(|x| x.usable_memory).collect();
    let mut placements: Vec<Option<Placement>> = vec![None; models.len()];
    for &m in order {
        let pm = &models[m];
        // The machines with memory left, each offering what is left.
        let idx: Vec<usize> = (0..n).filter(|&i| free[i].0 >= MIN_FREE).collect();
        let mut sub = cluster.clone();
        sub.nodes = idx
            .iter()
            .map(|&i| {
                let mut p = cluster.nodes[i].clone();
                if free[i] < p.usable_memory {
                    p.usable_memory = free[i];
                }
                p
            })
            .collect();
        let mut budgets = vec![Bytes(0); n];
        if sub.nodes.is_empty() {
            placements[m] = Some(Placement {
                plan: None,
                budgets,
                reason: Some("no machine has memory left".into()),
            });
            continue;
        }
        let r = plan(pm.spec, &sub, &pm.workload, &pm.opts);
        let mut selected = r.selected.clone();
        // Prefer staying on the current machines.
        let cur: Vec<usize> = idx
            .iter()
            .enumerate()
            .filter(|(_, full)| pm.current.contains(full))
            .map(|(k, _)| k)
            .collect();
        if !cur.is_empty() && cur.len() < sub.nodes.len() {
            let mut stay = sub.clone();
            stay.nodes = cur.iter().map(|&k| sub.nodes[k].clone()).collect();
            if let Some(mut p) = plan(pm.spec, &stay, &pm.workload, &pm.opts).selected {
                let best = selected.as_ref().map(|b| b.tokens_per_sec).unwrap_or(0.0);
                if best < p.tokens_per_sec * MOVE_GAIN {
                    for st in &mut p.stages {
                        st.node = cur[st.node];
                    }
                    selected = Some(p);
                }
            }
        }
        match selected {
            Some(mut p) => {
                for st in &mut p.stages {
                    let full = idx[st.node];
                    st.node = full;
                    let want = need(st.mem.peak, pm.opts.safety_frac).min(free[full]);
                    budgets[full] = want;
                    free[full] = free[full].saturating_sub(want);
                }
                placements[m] = Some(Placement {
                    plan: Some(p),
                    budgets,
                    reason: None,
                });
            }
            None => {
                let reason = r
                    .rejected
                    .first()
                    .map(|x| x.reason.clone())
                    .unwrap_or_else(|| "no feasible placement".into());
                placements[m] = Some(Placement {
                    plan: None,
                    budgets,
                    reason: Some(reason),
                });
            }
        }
    }
    Allocation {
        placements: placements.into_iter().map(|p| p.unwrap()).collect(),
        free,
    }
}

/// Higher is better: models placed, then earlier-listed models placed, then speed.
fn score(a: &Allocation) -> (usize, u64, f64) {
    let k = a.placements.len().min(63);
    let mut mask = 0u64;
    let mut tps = 0.0;
    for (i, p) in a.placements.iter().enumerate().take(k) {
        if let Some(plan) = &p.plan {
            mask |= 1 << (k - 1 - i);
            // Relative speed so a small fast model doesn't dominate the sum.
            tps += plan.tokens_per_sec.max(1e-9).ln();
        }
    }
    (a.placed(), mask, tps)
}

fn permutations(k: usize) -> Vec<Vec<usize>> {
    fn rec(cur: &mut Vec<usize>, used: &mut [bool], out: &mut Vec<Vec<usize>>) {
        if cur.len() == used.len() {
            out.push(cur.clone());
            return;
        }
        for i in 0..used.len() {
            if !used[i] {
                used[i] = true;
                cur.push(i);
                rec(cur, used, out);
                cur.pop();
                used[i] = false;
            }
        }
    }
    let mut out = Vec::new();
    rec(&mut Vec::new(), &mut vec![false; k], &mut out);
    out
}

/// Split `cluster` between `models`.
pub fn allocate(models: &[PoolModel], cluster: &Cluster) -> Allocation {
    let k = models.len();
    let orders = if k <= 5 {
        permutations(k)
    } else {
        // Listed order, and largest first (first-fit decreasing).
        let listed: Vec<usize> = (0..k).collect();
        let mut big = listed.clone();
        big.sort_by_key(|&i| std::cmp::Reverse(models[i].spec.weight_bytes()));
        vec![listed, big]
    };
    let mut best: Option<(Allocation, (usize, u64, f64))> = None;
    for o in orders {
        let a = place_in_order(models, cluster, &o);
        let s = score(&a);
        let better = match &best {
            None => true,
            Some((_, b)) => {
                (s.0, s.1) > (b.0, b.1) || ((s.0, s.1) == (b.0, b.1) && s.2 > b.2 + 1e-9)
            }
        };
        if better {
            best = Some((a, s));
        }
    }
    let (mut a, _) = best.expect("at least one order");
    // Explain unplaced models in terms of what the others took.
    let placed: Vec<String> = a
        .placements
        .iter()
        .zip(models)
        .filter(|(p, _)| p.plan.is_some())
        .map(|(_, m)| m.spec.id.clone())
        .collect();
    for (p, m) in a.placements.iter_mut().zip(models) {
        if p.plan.is_none() && !placed.is_empty() {
            let alone = plan(m.spec, cluster, &m.workload, &m.opts)
                .selected
                .is_some();
            if alone {
                p.reason = Some(format!(
                    "fits on its own, but not next to {} — add a machine (`tendril join`) or serve fewer models",
                    placed.join(", ")
                ));
            }
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::Link;
    use crate::model::catalog;
    use crate::presets::node_from_spec;

    fn cluster(nodes: &[(&str, &str)]) -> Cluster {
        Cluster::new(
            nodes
                .iter()
                .map(|(n, s)| node_from_spec(n, s).unwrap())
                .collect(),
            Link::preset("10gbe").unwrap(),
        )
    }

    fn pm(spec: &ModelSpec) -> PoolModel<'_> {
        PoolModel {
            spec,
            workload: Workload::new(4096, 2),
            opts: PlanOptions::default(),
            current: vec![],
        }
    }

    #[test]
    fn running_models_stay_put_for_small_gains() {
        let a = catalog::lookup("qwen2.5-0.5b").unwrap().spec();
        // Two identical machines: nothing to gain by moving.
        let c = cluster(&[("old", "m4:32"), ("new", "m4:32")]);
        let mut m = pm(&a);
        m.current = vec![0];
        let r = allocate(&[m], &c);
        let p = r.placements[0].plan.as_ref().unwrap();
        assert!(p.stages.iter().all(|s| s.node == 0), "{:?}", p.label());
        assert_eq!(r.placements[0].budgets[1], Bytes(0));
        // A much faster machine is worth moving to.
        let c = cluster(&[("old", "m4:32"), ("fast", "m4-max:128")]);
        let mut m = pm(&a);
        m.current = vec![0];
        let r = allocate(&[m], &c);
        let p = r.placements[0].plan.as_ref().unwrap();
        assert!(p.stages.iter().all(|s| s.node == 1), "{:?}", p.label());
    }

    #[test]
    fn two_small_models_share_one_machine() {
        let a = catalog::lookup("qwen2.5-0.5b").unwrap().spec();
        let b = catalog::lookup("llama-3.2-1b").unwrap().spec();
        let c = cluster(&[("mac", "m4:32")]);
        let r = allocate(&[pm(&a), pm(&b)], &c);
        assert_eq!(r.placed(), 2);
        let total: u64 = r.placements.iter().map(|p| p.budgets[0].0).sum();
        assert!(total <= c.nodes[0].usable_memory.0);
        // Each model re-planned alone inside its budget still fits.
        for (p, spec) in r.placements.iter().zip([&a, &b]) {
            let mut sub = c.clone();
            sub.nodes[0].usable_memory = p.budgets[0];
            let again = plan(spec, &sub, &Workload::new(4096, 2), &PlanOptions::default());
            assert!(again.selected.is_some(), "{:?}", again.rejected.first());
        }
    }

    #[test]
    fn big_model_gets_the_big_machine() {
        let big = catalog::lookup("llama-3.1-8b").unwrap().spec();
        let small = catalog::lookup("qwen2.5-0.5b").unwrap().spec();
        let c = cluster(&[("small", "m4:16"), ("big", "m4:48")]);
        // Listed small-first; the allocator still finds room for both.
        let r = allocate(&[pm(&small), pm(&big)], &c);
        assert_eq!(r.placed(), 2, "{:?}", r.placements);
        let bp = r.placements[1].plan.as_ref().unwrap();
        assert!(bp.stages.iter().any(|s| s.node == 1));
    }

    #[test]
    fn priority_decides_when_not_everything_fits() {
        let a = catalog::lookup("llama-3.1-8b").unwrap().spec();
        let b = catalog::lookup("llama-3.1-8b").unwrap().spec();
        let c = cluster(&[("mac", "m4:32")]);
        let r = allocate(&[pm(&a), pm(&b)], &c);
        assert_eq!(r.placed(), 1);
        assert!(
            r.placements[0].plan.is_some(),
            "the first listed model wins"
        );
        let why = r.placements[1].reason.as_deref().unwrap();
        assert!(why.contains("not next to"), "{why}");
    }
}
