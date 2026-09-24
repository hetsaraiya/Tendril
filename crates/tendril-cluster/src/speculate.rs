//! Speculative decoding: guess the next few tokens cheaply, let the pipeline
//! check them all in one pass. On a distributed pipeline each pass pays every
//! machine's compute plus every network hop, while checking k+1 tokens costs
//! about the same as producing one — so accepted guesses are nearly free.
//!
//! Drafts come from *prompt lookup*: the longest recent n-gram that already
//! appeared in the conversation proposes what followed it last time. It costs
//! nothing, needs no second model, and shines when outputs copy their inputs
//! (code edits, summaries, RAG, structured data). Verification is exact, so
//! the output distribution is unchanged; greedy output is identical.

/// Propose up to `k` tokens by finding the latest earlier occurrence of the
/// context's trailing n-gram (longest n first) and copying what followed.
pub fn lookup(ctx: &[u32], k: usize, max_ngram: usize) -> Vec<u32> {
    if k == 0 || ctx.len() < 2 {
        return Vec::new();
    }
    for n in (1..=max_ngram.min(ctx.len() - 1)).rev() {
        let tail = &ctx[ctx.len() - n..];
        // Search backwards, excluding the tail itself; cap the scan for long contexts.
        let hi = ctx.len() - n;
        let lo = hi.saturating_sub(16_384);
        for start in (lo..hi).rev() {
            if &ctx[start..start + n] == tail {
                let from = start + n;
                let end = (from + k).min(ctx.len());
                if from < end {
                    return ctx[from..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// Decides how many tokens to draft, learning from what gets accepted and
/// what a verification pass costs compared with a plain step.
#[derive(Clone, Debug)]
pub struct Controller {
    pub k: usize,
    pub max_k: usize,
    accept: f64,
    spec_ms: f64,
    plain_ms: f64,
    paused: usize,
}

impl Controller {
    pub fn new(max_k: usize) -> Controller {
        Controller {
            k: max_k.clamp(1, 4),
            max_k: max_k.max(1),
            accept: 0.6,
            spec_ms: 0.0,
            plain_ms: 0.0,
            paused: 0,
        }
    }

    /// Tokens to draft for the next step (0 = take a plain step).
    pub fn draft_len(&mut self) -> usize {
        if self.paused > 0 {
            self.paused -= 1;
            return 0;
        }
        self.k
    }

    pub fn on_plain_step(&mut self, ms: f64) {
        self.plain_ms = if self.plain_ms == 0.0 {
            ms
        } else {
            self.plain_ms * 0.9 + ms * 0.1
        };
    }

    pub fn on_spec_step(&mut self, ms: f64, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        let rate = accepted as f64 / drafted as f64;
        self.accept = self.accept * 0.8 + rate * 0.2;
        self.spec_ms = if self.spec_ms == 0.0 {
            ms
        } else {
            self.spec_ms * 0.8 + ms * 0.2
        };
        if accepted == drafted {
            self.k = (self.k + 1).min(self.max_k);
        } else if accepted == 0 {
            self.k = self.k.saturating_sub(1).max(1);
        }
        // Is speculating actually faster? Expected tokens per verify pass vs its cost.
        if self.plain_ms > 0.0 && self.spec_ms > 0.0 {
            let a = self.accept.clamp(0.0, 0.999);
            let expected = (1.0 - a.powi(self.k as i32 + 1)) / (1.0 - a);
            let gain = expected * self.plain_ms / self.spec_ms;
            if gain < 1.0 {
                // Losing: take plain steps for a while, then try again gently.
                self.paused = 24;
                self.k = 2;
            }
        }
    }

    pub fn acceptance(&self) -> f64 {
        self.accept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_copies_what_followed() {
        // "a b c d ... a b" -> propose "c d"
        let ctx = [1, 2, 3, 4, 9, 9, 1, 2];
        assert_eq!(lookup(&ctx, 2, 3), vec![3, 4]);
        // Longest n-gram wins over a more recent shorter match.
        let ctx = [5, 6, 7, 8, 1, 6, 7, 3, 5, 6, 7];
        assert_eq!(lookup(&ctx, 1, 3), vec![8]);
        assert!(lookup(&[1, 2, 3], 4, 3).is_empty());
        assert!(lookup(&[1, 2, 3, 1], 0, 3).is_empty());
    }

    #[test]
    fn controller_backs_off_when_losing() {
        let mut c = Controller::new(6);
        c.on_plain_step(10.0);
        for _ in 0..10 {
            let k = c.draft_len();
            if k > 0 {
                c.on_spec_step(30.0, k, 0); // expensive and never accepted
            }
        }
        assert_eq!(c.draft_len(), 0, "should pause speculation");
        let mut c = Controller::new(6);
        c.on_plain_step(10.0);
        for _ in 0..10 {
            let k = c.draft_len();
            c.on_spec_step(11.0, k, k);
        }
        assert_eq!(c.k, 6, "full acceptance grows the draft");
    }
}
