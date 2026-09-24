//! Prefix cache: finished conversations stay parked on the stages so the
//! next turn — which re-sends the whole conversation — only prefills what's
//! new. Parked KV lives in memory the plan already reserved (at most
//! `slots` live sequences, active or parked); beyond that, the least recently
//! used conversations spill to each machine's disk, or are dropped.

use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Entry {
    pub seq: u64,
    /// Tokens whose KV the stages hold (prompt + generated tokens fed back).
    pub tokens: Vec<u32>,
    pub on_disk: bool,
    pub last_used: Instant,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub seq: u64,
    /// Prompt tokens whose KV is reused.
    pub reuse: usize,
    /// Cut the parked KV back to `reuse` first (it holds a longer, divergent history).
    pub truncate: bool,
    /// The KV is on disk and must be restored first.
    pub restore: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Evict {
    /// Move to disk (still reusable).
    Spill(u64),
    /// Free memory; gone.
    Release(u64),
    /// Delete from disk; gone.
    DropDisk(u64),
}

pub struct PrefixCache {
    pub entries: Vec<Entry>,
    /// Sequences the planned KV memory can hold at once.
    pub slots: usize,
    /// Tokens of KV that may be parked on disk.
    pub disk_tokens: usize,
    /// Shortest prefix worth reusing.
    pub min_match: usize,
    /// Sliding-window size, if the model has one: longer histories can't be truncated.
    pub window: Option<usize>,
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl PrefixCache {
    pub fn new(slots: usize, disk_tokens: usize, window: Option<usize>) -> PrefixCache {
        PrefixCache {
            entries: Vec::new(),
            slots: slots.max(1),
            disk_tokens,
            min_match: 16,
            window,
        }
    }

    pub fn ram_entries(&self) -> usize {
        self.entries.iter().filter(|e| !e.on_disk).count()
    }

    pub fn disk_entries(&self) -> usize {
        self.entries.iter().filter(|e| e.on_disk).count()
    }

    /// Take the parked sequence sharing the longest prefix with `prompt`.
    pub fn lookup(&mut self, prompt: &[u32]) -> Option<Hit> {
        let mut best: Option<(usize, usize)> = None; // (index, reuse)
        for (i, e) in self.entries.iter().enumerate() {
            // Keep at least one prompt token to compute fresh logits from.
            let reuse = lcp(&e.tokens, prompt).min(prompt.len().saturating_sub(1));
            if reuse < self.min_match {
                continue;
            }
            let truncate = reuse < e.tokens.len();
            if truncate && self.window.is_some_and(|w| e.tokens.len() > w) {
                continue; // its start may have left the sliding-window cache
            }
            if best.is_none_or(|(_, r)| reuse > r) {
                best = Some((i, reuse));
            }
        }
        let (i, reuse) = best?;
        let e = self.entries.remove(i);
        Some(Hit {
            seq: e.seq,
            reuse,
            truncate: reuse < e.tokens.len(),
            restore: e.on_disk,
        })
    }

    /// Park a finished sequence.
    pub fn insert(&mut self, seq: u64, tokens: Vec<u32>) {
        self.entries.push(Entry {
            seq,
            tokens,
            on_disk: false,
            last_used: Instant::now(),
        });
    }

    /// Make sure `active` running sequences plus parked ones fit the plan's
    /// memory, and parked-on-disk ones fit the disk budget.
    pub fn enforce(&mut self, active: usize) -> Vec<Evict> {
        let mut out = Vec::new();
        while self.ram_entries() + active > self.slots {
            let Some(i) = self.lru(false) else { break };
            let disk_used: usize = self
                .entries
                .iter()
                .filter(|e| e.on_disk)
                .map(|e| e.tokens.len())
                .sum();
            if disk_used + self.entries[i].tokens.len() <= self.disk_tokens {
                self.entries[i].on_disk = true;
                out.push(Evict::Spill(self.entries[i].seq));
            } else {
                out.push(Evict::Release(self.entries.remove(i).seq));
            }
        }
        loop {
            let disk_used: usize = self
                .entries
                .iter()
                .filter(|e| e.on_disk)
                .map(|e| e.tokens.len())
                .sum();
            if disk_used <= self.disk_tokens {
                break;
            }
            let Some(i) = self.lru(true) else { break };
            out.push(Evict::DropDisk(self.entries.remove(i).seq));
        }
        out
    }

    fn lru(&self, on_disk: bool) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.on_disk == on_disk)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(i, _)| i)
    }

    /// Everything parked (for teardown).
    pub fn drain(&mut self) -> Vec<Entry> {
        std::mem::take(&mut self.entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_longest_prefix() {
        let mut c = PrefixCache::new(2, 0, None);
        let a: Vec<u32> = (0..40).collect();
        let mut b: Vec<u32> = (0..30).collect();
        b.extend([900, 901]);
        c.insert(1, a.clone());
        c.insert(2, b);
        let mut prompt = a.clone();
        prompt.extend([5, 6, 7]);
        let h = c.lookup(&prompt).unwrap();
        assert_eq!(
            h,
            Hit {
                seq: 1,
                reuse: 40,
                truncate: false,
                restore: false
            }
        );
        // Divergent history is truncated back to the shared part.
        let mut p2: Vec<u32> = (0..30).collect();
        p2.extend([1, 2, 3]);
        assert_eq!(
            c.lookup(&p2).unwrap(),
            Hit {
                seq: 2,
                reuse: 30,
                truncate: true,
                restore: false
            }
        );
        // Too short a match is not worth it.
        c.insert(3, (0..10).collect());
        assert!(c.lookup(&(0..20).collect::<Vec<_>>()).is_none());
        // Identical prompt keeps one token to recompute.
        c.insert(4, a.clone());
        assert_eq!(c.lookup(&a).unwrap().reuse, 39);
    }

    #[test]
    fn evicts_to_disk_then_drops() {
        let mut c = PrefixCache::new(2, 100, None);
        c.insert(1, vec![0; 60]);
        std::thread::sleep(std::time::Duration::from_millis(2));
        c.insert(2, vec![0; 60]);
        // One active request + two parked > 2 slots: oldest spills.
        assert_eq!(c.enforce(1), vec![Evict::Spill(1)]);
        std::thread::sleep(std::time::Duration::from_millis(2));
        c.insert(3, vec![0; 60]);
        // Disk budget 100 tokens: seq 2 can't spill (60 + 60 > 100) and is released.
        assert_eq!(c.enforce(1), vec![Evict::Release(2)]);
        assert_eq!(c.disk_entries(), 1);
        c.disk_tokens = 10;
        assert_eq!(c.enforce(0), vec![Evict::DropDisk(1)]);
    }

    #[test]
    fn sliding_window_guard() {
        let mut c = PrefixCache::new(2, 0, Some(32));
        let mut long: Vec<u32> = (0..50).collect();
        c.insert(1, long.clone());
        long.truncate(40);
        long.extend([7, 8]);
        assert!(
            c.lookup(&long).is_none(),
            "would need truncation beyond the window"
        );
    }
}
