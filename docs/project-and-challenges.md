# Tendril: the project and its hard problems

Status: proposed product direction; no inference implementation yet. This document clarifies the idea from the [original discussion](source-conversation.md). Runtime capabilities, API availability, hardware examples and performance claims in that discussion still need verification.

For the recommended experiment order, see [Approaches and experiments](approaches-and-experiments.md).

## 1. What is Tendril?

**Tendril discovers a pool of computers and chooses a practical way to run a requested LLM on them.**

It is an inference planner and execution coordinator, not a mechanism that turns arbitrary RAM into one giant GPU.

Given machines, a model and a workload, it should answer:

- Can this exact model run with the requested context length and concurrency?
- Which machines should execute it, and which should be left out?
- Which supported runtime, model representation and partition should be used?
- Where should weights and KV live?
- What latency, throughput and memory use should the user expect?
- Why was this plan chosen, and what alternatives were rejected?

The answer may be a single machine, a distributed pipeline, a recommendation to use a smaller representation, or an honest explanation that no suitable plan exists. Changing model precision/quality requires user permission; it is not a silent fallback.

## 2. Who and what is it for?

The initial user owns a few local machines but cannot comfortably fit a desired model on any one of them. They want a usable local inference endpoint without manually assigning layers.

The broader opportunity is heterogeneous hardware: Apple Silicon, NVIDIA GPUs and CPU-only machines with different memory, links and capabilities. Supporting that pool does **not** require every device to participate in every forward pass. Later, nodes might serve separate models or act as compute, draft-model, cache or storage workers—but only when that role is actually beneficial.

### Core promise

> Give Tendril machines, a supported model and workload constraints. It finds and explains a feasible execution plan, then runs it reliably.

### What it does not promise

- Adding computers always makes generation faster.
- Combined RAM behaves like one GPU's VRAM.
- Arbitrary models, quantizations and runtimes are interchangeable.
- A laptop can disappear without affecting an active request.
- Distributed execution is better than a smaller model on one machine.

## 3. First real target

Two Apple Silicon Macs with 16 GB unified memory each, running a supported dense model with roughly 19 GB of weights. The model's runtime footprint must prevent safe execution on either Mac alone while fitting across both at a declared context length and concurrency.

The exact checkpoint is **not selected**. The source conversation's Gemma example is not proof of architecture or runtime support.

Proposed execution:

```text
Prompt / token
      |
Mac A: embeddings + early layers + their hot KV
      |
      | boundary activations
      v
Mac B: later layers + their hot KV + final norm + output head
      |
      | sampled token returned to the start
      v
Next decode step
```

Weights remain resident at their assigned stages. The split is automatic, not necessarily equal. The planner must account for embeddings, the output head, tied weights and model-specific dependencies—not just divide the count of transformer layers.

### First-version success criteria

- Actual memory use fits each machine with declared safety headroom, including loading peaks.
- Each worker loads only the weights its partition needs.
- Distributed execution passes correctness checks against a supported reference.
- The planner evaluates legal cuts and explains its selected placement.
- A model too large for either node safely generates through one streaming, OpenAI-compatible endpoint.
- Cancellation, overload and node loss produce bounded cleanup and explicit errors.
- Measured latency and throughput are reported honestly; speedup is not required.

## 4. Product boundary

Rust is the proposed language for Tendril's planner, coordination and application logic. Accelerator execution should reuse proven runtime machinery where possible. Evaluate integration before building custom kernels or independent CUDA and MLX model implementations.

Logical responsibilities—not a commitment to a large crate hierarchy:

1. **Inventory:** node identity, capabilities, usable memory and measured links.
2. **Model inspection:** architecture, tensor sizes, runtime support and memory requirements.
3. **Planning:** feasible placements, workload cost and explanations.
4. **Execution:** runtime integration, partition loading, request routing and generation.
5. **Operations:** admission limits, authentication, metrics, cancellation and failures.

Existing execution engines may cover several of these responsibilities. Manual node configuration is enough for initial experiments; automatic discovery is not the first technical risk to solve.

## 5. The elephants in the room

### A. Autoregressive decoding is sequential

For an ordinary single conversation, the next token cannot start until the previous token's forward pass and sampling finish.

```text
Time per token ≈ sum(stage compute + boundary costs) + sampling/return cost
```

Putting successive layers on different machines provides capacity, not automatic parallel speedup for that sequence. Multiple independent requests can occupy different stages concurrently; throughput then depends heavily on bottleneck service time, scheduling and shared resources.

**Consequence:** interactive latency and aggregate throughput require different planner objectives. Balancing stage durations is not automatically optimal for one user.

### B. More resources can make the critical path slower

A CPU stage, thermally throttled laptop or unstable link can add more time than its memory capacity is worth. Nominal hardware specifications do not capture current contention or sustained performance.

**Consequence:** the planner must evaluate subsets of nodes, measure useful performance and be allowed to reject a node entirely. Assigning it a cache role is not automatically useful either.

### C. Network overhead is more than payload size

A small decode activation still incurs runtime synchronization, device-to-host copies where needed, framing, scheduling, network delay and receiver tensor preparation. Prefill transfers can be much larger. Jitter can harm streaming responsiveness even when average bandwidth looks good.

**Consequence:** measure the complete boundary, including the sampled-token return path. Do not infer inference performance from a bandwidth test alone.

### D. Memory is fragmented, dynamic and tiered

GPU VRAM, Apple unified memory, host RAM and SSD have different execution and access properties. Two 16 GB machines do not provide 32 GB of usable weight memory. KV grows with workload; loading, dequantization and temporary buffers can create peaks above steady-state usage.

**Consequence:** enforce per-node memory budgets for a declared context/concurrency. Reject unsafe plans rather than depending on OS swapping or accidental overcommit.

### E. Runtime compatibility is a model-execution problem

Moving tensor bytes is necessary but insufficient. Backends must agree on model revision, operators, positional encoding, attention semantics, tensor layouts, quantization and precision. A working CUDA-to-MLX transfer does not prove the model can be divided correctly between them.

**Consequence:** start with one supported runtime/model combination. Prefer an existing multi-backend execution engine before assembling unrelated runtimes. Maintain an explicit support matrix as coverage grows.

### F. KV state makes failures expensive

Each stage owns attention state accumulated from the conversation. Losing a worker loses both execution capacity and potentially its KV. Another machine needs compatible weights, capacity and either restored KV or re-prefill before taking over.

Independent asynchronous copies of KV are not necessarily a consistent distributed checkpoint. Recovery also needs a committed token position, compatible cache representation and replay information.

**Consequence:** clear request failure and restart is the first recovery policy. Transparent continuation is a separate feature, not a consequence of heartbeats.

### G. Quantization and speculation are trade-offs, not free fixes

A smaller representation might remove an entire node from the decode path, but can affect quality and runtime support. Speculation might amortize distributed verification costs, but adds draft latency, memory, rejected work and KV rollback complexity. Exact sampling requires a correct acceptance/rejection algorithm.

**Consequence:** benchmark total cost per accepted token and validate quality/correctness. Neither technique should be advertised as a guaranteed speedup.

### H. Untrusted peers and resource exhaustion

LAN membership is not authentication. Workers handling prompts, weights and KV are inside the data-trust boundary. Encryption protects data in transit, not from a participating worker that can read it. Malformed tensor headers, unlimited queues or oversized contexts can exhaust a node.

**Consequence:** v1 targets trusted, explicitly authorized machines; use authenticated/encrypted communication and bounded resources. A public volunteer pool is out of scope. Checksums detect some corruption but do not prove a malicious worker computed correctly.

### I. Prior art and product value

The original discussion identifies exo, llama.cpp RPC, Petals and vLLM as relevant references. Their current support and integration constraints need direct verification. Merely demonstrating distributed inference does not establish a new product advantage.

**Consequence:** compare with existing tools. Tendril earns its complexity through demonstrably better placement, setup, explanations or reliability—not by rebuilding tensor transport.

## 6. Initial non-goals

- Universal CUDA + MLX + CPU execution in one pipeline.
- Internet-scale, untrusted volunteer inference.
- Custom accelerator kernels or a custom RDMA stack.
- Remote SSD on the normal per-token hot-KV path.
- Automatic failover with no request interruption.
- Multi-coordinator consensus and multi-tenant scheduling.
- Every possible model architecture or quantization.

These are scope boundaries, not claims that the problems can never be solved.

## 7. Questions experiments must settle

1. Which existing engine can execute the selected model and expose enough placement control?
2. Does the oversized target fit under measured peak memory at a useful context length?
3. Is the resulting generation speed acceptable to the intended user?
4. How closely can a simple planner predict the best measured two-node placement?
5. Is the distributed target worth using compared with a permitted single-node alternative?
6. Is there enough value beyond existing tools to justify owning this software?
