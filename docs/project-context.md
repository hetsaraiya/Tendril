# Tendril — imported project context

Source: [shared ChatGPT conversation](https://chatgpt.com/share/6ab3f3f3-1efc-83e9-acba-bb50eccb5708). The [full readable transcript](source-conversation.md) preserves the original discussion and 100-section Rust master prompt (beginning around line 1012). This document is a digest of *ideas*, not proof that any feature exists or that a design has been approved.

## Product idea

Make a pool of computers into an intelligently scheduled LLM inference fabric. Discover node compute, memory, networking and storage; inspect a requested model; choose a viable runtime, placement, partition and cache strategy; explain the choice; then serve inference. The value proposition is the **planner**: decide when using a node helps, when it only makes an otherwise impossible model fit, and when it would hurt latency. Do not sell pooled RAM as one giant GPU.

The repository is named **Tendril**. The chat also proposed *Tendril* among nature-inspired names, but used **AetherMesh** as a *placeholder* throughout its master build prompt. Treat `aether`, `aetherd`, etc. as provisional example names, not the repository's settled branding.

## First concrete target

- Two Apple Silicon machines (example: M4 and M5), 16 GB unified memory each; a dense MLX-format model of roughly 19 GB of weights that cannot fit safely on either one alone. Gemma was suggested as an example, not a verified/selected checkpoint.
- Inspect actual tensors and runtime support before assuming model architecture, size, layer count, quantization or MLX compatibility.
- Automatically choose a **contiguous layer/pipeline split**: first node runs embeddings and early layers, second runs later layers, norm and output head. Keep only each node's assigned weights resident; transfer boundary activations, not weights, for every token. Keep each stage's hot KV with the attention layers it computes.
- Account for available memory after OS/runtime reservation, weights, KV at requested context/concurrency, activation buffers and safety margins. Profile per-layer compute and network RTT/bandwidth. For two nodes evaluate every legal cut (and both node orders), reject non-fitting placements, and score feasible plans for the chosen workload. Explain rejected plans and estimated cost. A 50/50 split is not a default.
- Success means actually serving correct generated tokens through an OpenAI-compatible endpoint from a model that cannot safely run on either Mac alone, with no user-specified layer placement. **Fitting is the v1 goal; speedup is not promised.**

## Suggested build order from the conversation

1. Rust CLI/daemon skeleton, identities, configuration, logging and clean shutdown; two daemons can see each other.
2. Manual join/discovery, authenticated membership, heartbeats and capability exchange; measured network/hardware topology.
3. Dense model manifest/inspector (e.g. MLX/SafeTensors), tensor sizes and runtime capability validation without loading all weights.
4. Memory estimator and two-node contiguous placement planner; simulator/fake runtime for deterministic tests of planning, transport, cancellation and failures.
5. Verify supported MLX model inference on **one** node through a Rust adapter/isolated FFI, then partition loading (prove no node loads the entire checkpoint).
6. Two-Mac prefill/decode pipeline with local per-layer KV ownership, activation transfer and streaming API.
7. Bounded concurrency, cancellation, failure cleanup, profiling/benchmarking; add cache tiers and advanced strategies only after the above is real.

The source prompt has much broader milestone and edge-case lists; consult it when implementing a specific feature, rather than scaffolding every proposed crate/interface up front. Rust owns the application/control plane; MLX/Metal execution can be via Rust bindings or narrowly scoped `mlx-c` FFI. No Python production service was proposed.

## Technical boundaries and risks

- **Serial decode:** a single conversation needs a full forward pass before the next token; pipeline stages and network boundaries add latency. Multiple independent requests can fill pipeline bubbles and improve throughput. Do not assume balancing stage times minimizes *single-request* token latency; score end-to-end latency versus multi-request throughput explicitly.
- **Heterogeneous nodes:** CUDA, Apple/MLX and CPU have incompatible execution engines and often incompatible weight/quantization layouts. Prefer compatible compute islands; a CPU or slow laptop may be better used for cold storage, routing or a small draft model than on the decode path. Mixed-runtime pipelines are a *later* capability, not v1.
- **Cross-runtime tensor transfer:** the discussed candidate path is CUDA device → pinned host RAM → versioned/validated wire tensor (shape, dtype, byte length, request/plan identifiers) → network → MLX/CPU tensor; reverse similarly. NCCL is for NVIDIA-compatible groups; MLX Distributed for Apple-compatible groups. Neither provides direct NVIDIA↔Apple conversion. QUIC/Quinn was proposed as a transport candidate; verify APIs and benchmark against simpler options before committing.
- **KV hierarchy:** active KV stays near compute (accelerator/unified memory, then local RAM). SSD/remote RAM is for cold/paused KV, reusable prefixes, spill, or checkpoints; pulling hot KV over the network each token is likely disastrous. Node loss can mean re-prefill unless KV can be restored/replayed.
- **Security/correctness:** untrusted peers need authentication and encrypted links; validate dimensions/lengths before allocation, limit queues/resources, use plan epochs and request IDs to reject stale/duplicate/out-of-order work, clean up on cancellation, surface failed requests rather than silently corrupting tokens. Source includes detailed failure/edge-case lists.
- **Later optimizations, not initial requirements:** quantization-aware scheduling, MLX tensor parallelism on sufficiently fast links, CUDA/CPU/ROCm backends, runtime islands, prefix caching, prefill/decode disaggregation, speculative draft/verify, asynchronous KV checkpointing and multi-request microbatching.

## Existing systems and validation needed

The conversation calls out **exo**, **llama.cpp RPC**, **Petals** and **vLLM** as prior art; compare their actual current behavior before rebuilding functionality or claiming novelty. It includes research assertions, concrete API sketches and opaque ChatGPT citation tokens; these are **not independently verified here**. In particular validate the exact target checkpoint/license/model compatibility, Rust↔MLX bindings, hardware constraints, measured memory and throughput, network transport, and availability of speculative-decoding APIs before writing a design around them.

## Source map

- Early feasibility and two-Mac walkthrough: transcript lines 9–534.
- Dense-model cut planning: lines 535–919.
- Naming (including Tendril) and 100-section engineering prompt: lines 920–4197.
- NVIDIA + Apple + x86 example: lines 4198–4775.
- Proposed CUDA/MLX/CPU transfer APIs and tensor-wire design: lines 4776–5644.
- Performance elephant, existing systems and product thesis: lines 5645–6202.
- Speculation, decode/prefill roles and other longer-term remedies: lines 6203–end.
