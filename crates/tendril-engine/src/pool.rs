//! A small spinning thread pool for the matmul kernels.
//!
//! Decode issues a few hundred short matmuls per token. Waking sleeping
//! threads for each one costs more than the work; these workers spin for a
//! moment after each job so the next one starts in microseconds, then park.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

type Job = dyn Fn(usize) + Sync;

struct Shared {
    gen: AtomicU64,
    job: UnsafeCell<Option<*const Job>>,
    next: AtomicUsize,
    total: AtomicUsize,
    pending: AtomicUsize,
    sleepers: AtomicUsize,
    park: Mutex<()>,
    wake: Condvar,
}

// SAFETY: `job` is only written by the submitting thread while no job is
// running, and only read by workers between the generation bump and their
// `pending` decrement, during which the submitter blocks.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    workers: usize,
    submit: Mutex<()>,
}

const SPIN: Duration = Duration::from_micros(1500);

impl Pool {
    fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            gen: AtomicU64::new(0),
            job: UnsafeCell::new(None),
            next: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            park: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = threads.saturating_sub(1);
        for i in 0..workers {
            let s = shared.clone();
            std::thread::Builder::new()
                .name(format!("tendril-kernel-{i}"))
                .spawn(move || worker(s))
                .expect("spawn kernel thread");
        }
        Pool {
            shared,
            workers,
            submit: Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.workers + 1
    }

    /// Run `f(i)` for i in 0..n across all threads; returns when all are done.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if self.workers == 0 || n == 1 {
            (0..n).for_each(f);
            return;
        }
        let _g = self.submit.lock().unwrap_or_else(|e| e.into_inner());
        let s = &self.shared;
        // SAFETY: see `Shared`; we erase the lifetime but block until every
        // worker has finished with the pointer.
        unsafe {
            let ptr: *const Job =
                std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f);
            *s.job.get() = Some(ptr);
        }
        s.next.store(0, Ordering::Relaxed);
        s.total.store(n, Ordering::Relaxed);
        s.pending.store(self.workers, Ordering::Relaxed);
        s.gen.fetch_add(1, Ordering::Release);
        if s.sleepers.load(Ordering::Acquire) > 0 {
            let _l = s.park.lock().unwrap_or_else(|e| e.into_inner());
            s.wake.notify_all();
        }
        drain(s, f);
        while s.pending.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
        unsafe {
            *s.job.get() = None;
        }
    }
}

fn drain(s: &Shared, f: &(dyn Fn(usize) + Sync)) {
    let total = s.total.load(Ordering::Relaxed);
    loop {
        let i = s.next.fetch_add(1, Ordering::Relaxed);
        if i >= total {
            break;
        }
        f(i);
    }
}

fn worker(s: Arc<Shared>) {
    let mut seen = 0u64;
    loop {
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            let g = s.gen.load(Ordering::Acquire);
            if g != seen {
                seen = g;
                break;
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(256) && start.elapsed() > SPIN {
                let guard = s.park.lock().unwrap_or_else(|e| e.into_inner());
                s.sleepers.fetch_add(1, Ordering::AcqRel);
                let guard = if s.gen.load(Ordering::Acquire) == seen {
                    s.wake
                        .wait_timeout(guard, Duration::from_millis(250))
                        .unwrap_or_else(|e| e.into_inner())
                        .0
                } else {
                    guard
                };
                s.sleepers.fetch_sub(1, Ordering::AcqRel);
                drop(guard);
            } else if spins.is_multiple_of(16) {
                // Let other runnable threads (e.g. elementwise ops) use the core.
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: the submitter keeps the job alive until `pending` hits zero.
        let job = unsafe { (*s.job.get()).map(|p| &*p) };
        if let Some(f) = job {
            drain(&s, f);
        }
        s.pending.fetch_sub(1, Ordering::Release);
    }
}

pub fn global() -> &'static Pool {
    static P: OnceLock<Pool> = OnceLock::new();
    P.get_or_init(|| {
        let n = std::env::var("TENDRIL_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });
        Pool::new(n.max(1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn runs_every_index_once() {
        let p = Pool::new(4);
        for n in [1usize, 3, 17, 1000] {
            let sum = AtomicU64::new(0);
            let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            p.run(n, &|i| {
                sum.fetch_add(i as u64, Ordering::Relaxed);
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(sum.load(Ordering::Relaxed), (n as u64) * (n as u64 - 1) / 2);
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
        // After parking, jobs still run.
        std::thread::sleep(Duration::from_millis(10));
        let c = AtomicU64::new(0);
        p.run(10, &|_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(c.load(Ordering::Relaxed), 10);
    }
}
