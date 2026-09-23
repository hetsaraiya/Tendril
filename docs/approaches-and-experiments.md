# Tendril: approaches to try, in priority order

Read [Project and challenges](project-and-challenges.md) for scope. This is an experiment plan, not a promise that every optimization will work. It refines the older [imported context](project-context.md): **prove execution and measure bottlenecks before building extensive cluster infrastructure.** No experiments listed below have been run yet.

## 1. Operating rules

- Reuse an execution engine before implementing an inference runtime.
- Change one major variable at a time and retain a reference baseline.
- Establish correctness before reporting speed.
- Optimize interactive latency and multi-request throughput separately.
- Start with two nodes and exhaustive legal cuts; do not begin with a general graph optimizer.
- Leave a node unused if using it does not improve the chosen objective.
- Agree on acceptable latency, context length, quality and memory headroom before judging success. There is no honest universal tokens-per-second target.

## 2. Order at a glance

| Priority | Try | What it establishes |
|---|---|---|
| P0 | Existing-engine feasibility spike | Whether Tendril needs a new execution layer at all |
| P0 | Small model: one Mac versus the same model split across two | Correctness and the real distribution penalty |
| P0 | Oversized model on two Macs | The initial capacity value proposition |
| P1 | Measured two-node placement and admission control | Whether automatic planning adds value |
| P1 | Reduce boundaries and staging overhead | Whether avoidable transfer costs dominate |
| P1 | Supported single-node/quantized alternatives | Whether distributing is actually the best option |
| P1 | Failure, cancellation and security tests | Whether the minimum product behaves safely |
| P2 | Multi-request scheduling and prefix caching | Throughput and repeated-prompt improvements |
| P2 | Conventional speculative decoding | Whether verification amortizes decode overhead |
| P3 | Mixed hardware, native collectives, KV recovery, phase disaggregation | Extensions justified by measurements |
| Research | Speculative pipeline scheduling | A possible later attack on single-sequence pipeline bubbles |

P0–P1 are the first product investigation. P2 and beyond are conditional, not a mandatory backlog.

## 3. Establish one repeatable measurement protocol

For every run, record hardware, power/thermal conditions, OS, runtime version/commit, model revision, tokenizer, representation, context length, sampling settings, request concurrency and link type. Pin versions where possible.

Use a small fixed prompt suite covering short and longer prefill, enough output tokens to measure steady decode, and explicit warm/cold runs. Repeat runs and report sample counts and variability; do not infer a stable p95 from a handful of samples.

Measure:

- Time to first token, including a clear definition of queue/load time inclusion.
- Inter-token latency (including tails with sufficient observations) and output tokens/second.
- Aggregate throughput and per-request latency at each concurrency level.
- Per-stage compute, runtime synchronization, transfer/staging and sampling time.
- Peak and steady memory, swap activity, KV growth and loading peaks per node.
- Network bytes, boundary count, errors, cancellations and recovery time.
- Correctness against a reference and, for changed precision/model, a declared quality evaluation.

**Correctness check:** for a small supported model, compare intermediate boundary activations and teacher-forced next-token logits against a single-node reference using declared numerical tolerances. Check greedy generation too, but do not use exact generated-text equality as the only test: minor numerical differences can change the entire continuation. Investigate differences rather than assuming they are harmless.

Keep the benchmark inputs and runnable checks so future changes can be compared. Synthetic transfer tests supplement, not replace, end-to-end inference.

## 4. P0 — prove the premise before building the product

### Experiment A: reuse an existing engine

**Question:** can an existing engine execute the desired partition and expose enough control for Tendril?

Investigate exo and llama.cpp RPC as candidates; verify current support directly rather than relying on claims in the source conversation. For each viable candidate, record:

- Exact supported checkpoint/format, platforms, license and installation requirements.
- Automatic/manual placement controls and runtime memory behavior.
- Interfaces for loading, serving, cancellation and metrics.
- Whether stage-specific weight loading and KV ownership work as needed.
- Security limitations and whether integration fits the intended deployment.
- What Rust integration requires: a supported API, library/FFI, or another boundary.

If a tool is useful only as a benchmark because it conflicts with production constraints, say so. Do not assume every candidate satisfies the Rust/no-Python-production-service preference.

**Keep:** the smallest integration that meets the target and exposes required controls.

**Escalate:** write a narrow missing adapter or runtime feature only after documenting the concrete blocker. Custom CUDA kernels and separate model implementations are not the default fallback. Do not expose an insecure experimental backend on an untrusted network.

### Experiment B: measure the two-node penalty with a small model

Use a model that fits on one Mac. Run the same revision, tokenizer and precision:

1. On Mac A alone.
2. On Mac B alone.
3. Split across both, with a few legal cuts and both orders where supported.

Keep context, sampling and workload constant. Compare activations/logits before timing. Trace the activation boundary and sampled-token return path.

**Pass:** correct execution, understood memory use, and a breakdown of where the distributed run spends its time. A distributed slowdown is valid evidence, not experiment failure.

**Stop and fix:** unexplained numerical divergence, hidden full-model duplication, swap-dependent operation or unbounded memory growth.

### Experiment C: demonstrate the oversized target

Choose a verified supported dense checkpoint whose required runtime memory prevents safe single-node execution under the declared workload. Do not intentionally crash a machine to establish this; use model inspection, measurements and safe admission limits.

Load only each worker's assigned components and measure loading peaks as well as steady memory. Start at concurrency one and a modest declared context; increase context only within the memory budget.

**Pass:** useful generation within agreed limits, no reliance on heavy swapping, and clear capacity advantage over either node alone.

If the target does not fit or is unusably slow, try a smaller supported representation/context with explicit user approval—or revise the target. Do not disguise that change as success for the original configuration.

## 5. P1 — the simplest useful planner

### Experiment D: exhaustive two-node placement

For every legal contiguous split, evaluate both node orders, plus supported single-node plans. Include embeddings, output head, shared tensors and any model-specific boundary restrictions.

A candidate is feasible only when:

```text
weights + workload KV + peak buffers + runtime overhead + safety reserve
    <= allowed memory on each node
```

Treat unavailable model/operator/dtype support as infeasible even when the bytes fit. Use observed runtime memory to calibrate estimates; serialized weight size is not a reliable complete footprint.

For interactive decode, estimate:

```text
latency = stage A compute
        + full A→B activation boundary
        + stage B compute
        + sampling and token return
        + orchestration overhead
```

Use profiles at representative context lengths because attention and memory behavior change with context. For a multi-request objective, model bottleneck service time and validate throughput under actual concurrency instead of reusing the single-request score.

Compare the predicted winner with measured feasible placements on the two-node testbed. Record prediction error and how much slower the selected plan is than the measured best plan.

**Keep:** a simple calibrated estimator if it selects acceptably close to the measured best. Exhaustive search over legal cuts is enough here.

**Do not:** claim stage balancing minimizes interactive latency, or add a general optimizer before the simple one fails.

### Experiment E: minimize avoidable transfer overhead

First measure where the boundary time goes. Then try, in order:

1. Fewer stage boundaries and contiguous ownership.
2. A stable wired link available on the actual machines; compare with the existing link.
3. Reused buffers and fewer unnecessary copies, allocations and tensor conversions.
4. Runtime-native transfer APIs where they are compatible and supported.
5. If Tendril must own transport, a bounded, authenticated/encrypted baseline; compare alternatives such as TCP/TLS and QUIC using realistic payloads and concurrency.

For a future CUDA path, verify that asynchronous device-to-host transfers complete before network reads and that receive buffers remain valid until host-to-device operations finish. Pinned-buffer reuse is useful only with correct lifetimes and synchronization.

Measure small decode and larger prefill transfers separately, including tail latency and contention. Keep an optimization only if end-to-end results improve without breaking correctness, security or memory limits.

RDMA is a later option requiring verified hardware/OS/runtime support and evidence that communication remains the bottleneck. QUIC is a candidate, not an automatic performance winner.

### Experiment F: reduce the number of necessary nodes

Compare the distributed target with supported smaller representations that might fit on the faster node alone. Keep the original target as a separate result.

Record quality, context capacity, peak memory and end-to-end performance. Include a smaller model as an explicit product alternative when useful, not as an equivalent execution of the original model.

**Keep:** user-approved representations that meet quality constraints and improve the chosen objective. No silent quantization, model replacement or CPU fallback.

### Experiment G: baseline operational safety

Before calling the result a usable product, test:

- Killing either worker during loading, prefill and decode.
- Disconnecting the link, delayed messages and stale plan epochs.
- Cancelling a request while transfers or compute are pending.
- Contexts larger than admission limits and too many concurrent requests.
- Malformed lengths/dimensions, duplicate work and unauthorized joins.
- Failed model downloads, disk exhaustion and corrupt artifacts.

Use explicit request identity, plan/version ownership, deadlines and bounded queues. Reject invalid dimensions and sizes before allocating. Authenticate authorized peers; encrypt traffic and avoid logging prompt/KV contents by default.

**Initial recovery:** report failure clearly, release reservations, and offer an explicit restart after resources return. Do not silently stitch restarted output onto an already streamed answer. Keep re-prefill/restart behavior deterministic enough to test, without promising bitwise reproducibility across backends.

**Pass:** no silent corruption, cross-request state leakage, indefinite hangs or unbounded retained resources. Recovery time is measured rather than assumed.

## 6. P2 — try only after the baseline works

### H. Multiple independent requests and bounded batching

Interleave independent sequences to fill pipeline idle time; use runtime-supported batching rather than inventing new kernels. Sweep concurrency and batch limits while tracking per-request latency and KV memory.

**Useful when:** serving several simultaneous users.

**Keep when:** throughput improves while latency and memory remain within declared limits. This does not fix single-user serial decoding. Account for fairness so large prefills do not indefinitely stall existing decode requests.

### I. Prefix reuse, then cold cache tiers

Start with local reusable prefix state if the runtime supports it. Key reuse by exact model revision, tokenizer/template and token prefix, plus all representation/runtime settings affecting KV validity. Apply authorization/isolation to cache access.

Measure avoided prefill against lookup, restore and memory costs. Only then compare remote RAM or SSD storage under workloads with real prefix reuse.

**Keep when:** saved computation exceeds retrieval/transfer cost at a measured hit rate. Evict safely under pressure. Never assume that because a laptop has an SSD it should join the cache path.

### J. Conventional speculative decoding

Use an existing supported implementation first. Establish local draft/verify correctness before trying a remote draft worker or distributed target verification.

Measure:

```text
average time per emitted token =
    total drafting + communication + verification + rollback/bookkeeping time
    divided by total emitted tokens
```

Evaluate several draft lengths and representative prompts, recording acceptance, draft latency, target verification time and extra memory. Use the appropriate acceptance/rejection and correction procedure for the sampling configuration; “keep matching tokens” is not a general exact-sampling algorithm.

**Keep when:** end-to-end latency improves over ordinary decoding at the same intended target distribution/quality. Turn it off for workloads where it loses. A smaller model on a slow CPU is not automatically a fast draft model.

## 7. P3 — larger extensions with explicit gates

### K. Heterogeneous compute: shared runtime before cross-runtime bridges

First evaluate whether a proven common execution engine can cover the desired CPU/Metal/CUDA combination and model. It may avoid implementing model semantics twice.

If a custom cross-runtime pipeline is genuinely necessary, validate one model, representation and boundary at a time. Specify dtype, shape, layout, byte order, bounded lengths, request/plan identity and token position. Test reference activations/logits and KV behavior, not just byte round-trips.

Group compatible devices into contiguous compute islands to reduce conversions. Compare with leaving slower devices out or using them for independent inference. Do not add a backend until it improves a concrete workload or enables a valuable otherwise-infeasible one.

### L. Native tensor parallelism and faster interconnects

Try runtime-supported tensor parallelism only after verifying the actual interconnect and profiling communication. Compare against the best pipeline and single-node plans at identical model/workload settings.

**Gate:** measured end-to-end benefit after collectives, synchronization and memory overhead. Do not start with mixed CUDA/MLX tensor parallelism or custom collective implementations.

### M. Checkpoint/replay, then replicas if justified

Start with a committed-token log and measured full re-prefill recovery. If recovery is too expensive, test consistent KV checkpoints:

- All stage snapshots represent the same committed sequence position.
- Model revision, plan, tokenizer and cache representation are identified.
- Required sampling/RNG state is preserved for the declared replay semantics.
- Partial checkpoints are not exposed as complete; integrity is validated.
- Previously streamed tokens are not emitted twice after recovery.

Measure checkpoint bandwidth, memory, steady-state interference and recovery time at several intervals. Async copying is not free, even when it avoids directly blocking a token.

Stage replicas require enough spare capacity and compatible state; test them only when availability goals justify that cost. Checkpointing alone does not supply replacement hardware.

### N. Separate prefill and decode placement

First tune scheduling/chunked prefill within one compatible runtime. Separate pools only if prefill interferes materially with decode and there is sufficient capacity for weights and compatible KV on both sides.

Include queueing, KV transfer/conversion and duplicate weight residency in the comparison. Do not assume BF16-prefill KV can safely feed a differently quantized decode engine; validate compatibility and output semantics explicitly.

**Gate:** improved workload-level latency/throughput after state-movement costs. This is more likely a serving optimization than a first two-laptop feature.

### O. Research: speculative pipeline scheduling

Only after ordinary distributed inference and conventional speculation are measured, investigate techniques that keep multiple speculative candidates in different pipeline stages. Verify current papers/code and reproduce a baseline first.

Require a precise acceptance/correction scheme, bounded speculative state, rollback semantics and a comparison with conventional speculation. Treat this as research risk, not a committed v1 feature or an established speedup.

## 8. Decision gates

1. **Existing tools already solve the use case well?** Integrate them; focus Tendril on the remaining placement or usability gap.
2. **Same-model split is incorrect?** Fix execution before scheduling optimizations.
3. **Oversized target cannot fit safely?** Revisit representation, context or hardware; stop feature expansion.
4. **Performance is unusable even with the best measured placement?** Identify the dominant cost and compare simpler alternatives before building a general cluster manager.
5. **Planner is inaccurate?** Improve measurement and memory accounting before introducing sophisticated search.
6. **An advanced optimization loses?** Disable or remove it. More features are not evidence of progress.

## 9. First deliverables

The first useful outcome is a small evidence package, not a large Rust workspace:

- A verified model/runtime choice and documented integration blockers.
- Reproducible one-node/two-node correctness and performance runs.
- An oversized-model feasibility result with measured memory and workload limits.
- A table of legal cuts, predicted versus actual costs, and the chosen plan.
- Basic cancellation/failure checks and known security limitations.
- A short go/no-go decision explaining what Tendril adds beyond the chosen engine.

Build the minimum Rust planner and orchestration needed to reproduce that result. Expand only where these measurements identify a real gap.
