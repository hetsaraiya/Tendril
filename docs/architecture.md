# Tendril architecture

How the pieces fit, what crosses the network and why.

## Crates

| Crate | Role |
|---|---|
| `tendril-core` | Hardware profiles, model inspection (config.json, safetensors headers, GGUF, HuggingFace over range requests), the memory model and the placement planner. No execution. |
| `tendril-engine` | Executes a *stage*: a contiguous slice of a decoder-only transformer (embeddings? + layers [a, b) + head?). Tokenizer, chat templates, sampling. |
| `tendril-cluster` | Encrypted transport, the agent (`tendril join`), the coordinator (`tendril serve`), weight shards, the OpenAI-compatible HTTP API and the web UI. |
| `tendril-cli` | The `tendril` binary. |

## Life of a cluster

```text
tendril serve M            tendril join C --token T       tendril join C --token T
┌───────────────┐          ┌─────────────────┐            ┌─────────────────┐
│ coordinator   │◀─control─│ agent (m5-air)  │            │ agent (desktop) │
│  + local stage│          └─────────────────┘            └─────────────────┘
└───────────────┘
 1. Agents connect to the control port (7420), prove the token (Noise NNpsk0) and send
    their hardware profile. The coordinator measures each link (ping RTT, 4 MiB probe).
 2. The planner runs over the live machines: exact DP over contiguous layer cuts for
    every ordered subset, honest per-machine memory budgets. No fit → "waiting" status
    with advice; each join/leave triggers a re-plan.
 3. Loading: each agent receives `LoadStage` with the exact tensor list for its layers.
    It asks for the tensors it hasn't cached; the coordinator streams them (4 MiB
    chunks, bounded queue). The agent writes a standard safetensors *shard* and loads it.
 4. Data plane: stage i connects to stage i+1's data port; the last stage connects back
    to the coordinator. The first stage is fed by the coordinator.
```

## Per-token data flow

```text
coordinator ──tokens──▶ stage 0 ──hidden──▶ stage 1 ──hidden──▶ … ──▶ last stage
     ▲                                                                  │ samples
     └──────────────────────────── token id ────────────────────────────┘
```

- Prefill is split into chunks (≤512 tokens, fewer for long contexts so attention
  scratch stays under ~256 MiB). Chunks are sent back to back and pipeline through the
  stages; only the final chunk computes logits.
- The last stage **samples** (temperature/top-k/top-p/min-p/penalties, seeded), so a
  single token id — not a vocabulary-sized logits vector — returns per step.
- Each stage owns the KV cache of its layers for every sequence. `Release` travels the
  chain to free it when a request ends, is cancelled or fails.
- Every message carries the plan *epoch*; work from an old plan is dropped. Positions
  are checked, so duplicated or reordered work fails loudly instead of corrupting KV.
- Errors travel down the chain to the coordinator and end the request with a message.

## Discovery and recovery

- **Discovery:** the coordinator broadcasts a UDP beacon (port 7419) every second with
  the model, control port, machine count, state and a fingerprint (hash) of the cluster
  token. `tendril join --token T` listens for the beacon whose fingerprint matches.
- **Recovery:** every request keeps a log of committed tokens (the KV it has built).
  When a machine in the pipeline disconnects, the coordinator tears the plan down and
  sends each request a `Lost` notice; the request waits for a new plan (new epoch),
  replays its committed tokens in ≤64-token chunks (the same kernels as decoding, so the
  rebuilt KV is identical on CPU) and continues sampling. Each request only accepts
  results stamped with the epoch it currently runs on.

## Model pools

`tendril serve A B C` starts one coordinator per model. Each has its own pipeline,
batching, prefix cache and recovery. A `Pool` sits in front of them:

- **Control port:** the pool accepts every agent connection. It reads the `Hello`, which
  names a model and carries the agent's random per-process **machine id**, and hands the
  session to that model's coordinator. `Welcome` lists every model, and the agent opens
  one session per model (each with its own data port and stage).
- **Allocation:** when any model's machines change, the pool merges what every model
  sees into one cluster (machines keyed by machine id, links from whichever model
  measured them). It then runs `tendril_core::pool::allocate`:
  - every placement order is tried (listed order and largest-first when there are more
    than five models);
  - in each order, a model is planned with the ordinary planner on the memory the
    previous models left;
  - its **budget** on a machine is its stage's peak ÷ (1 − safety) plus 32 MiB, so
    re-planning inside the budget finds the same split;
  - the winner places the most models, then prefers earlier-listed models, then the
    highest summed log-throughput;
  - a model already running stays on its machines unless moving is ≥25% faster.
- **Planning inside a share:** a coordinator's cluster snapshot drops machines where its
  budget is zero and caps `usable_memory` at the budget elsewhere. If the running
  pipeline no longer fits its budget, the coordinator tears it down (requests resume via
  recovery). Budgets that shrink are applied 1.5 s before budgets that grow, so memory
  is freed before another model loads into it.
- **Events** carry an optional `model` tag. Machine join and leave events come only from
  the first model, so a machine is reported once.

## Security

- One cluster token (generated on first `serve`, stored with 0600 permissions). Every
  TCP connection performs a Noise `NNpsk0_25519_ChaChaPoly_BLAKE2s` handshake keyed by
  the token: unauthenticated peers can't join, read activations or inject work.
- Message sizes are bounded (512 MiB) and validated before allocation; tensor shapes are
  checked against byte lengths.
- Machines in a cluster see the prompts and activations they process — only join
  machines you trust. Encryption protects the network, not a participant.

## Engine

- One generic decoder covers Llama, Mistral, Qwen2/3, Gemma 1/2/3 and Phi-3 through
  configuration flags (norm style, qk-norm, sandwich norms, soft-capping, sliding
  windows, local/global RoPE, fused projections). Verified against `transformers`.
- CPU: hand-written AVX2/F16C (and auto-vectorized NEON) GEMV kernels read weights in
  their stored precision (bf16/f16/f32) or q8_0 and run on a spinning thread pool;
  decode runs at DRAM bandwidth. Prefill widens weights in bands and uses a blocked GEMM.
- GPU (Metal/CUDA): candle kernels in native bf16/f16; optional on-load quantization.
- Q/K/V and gate/up projections are fused into single matmuls.
- Sliding-window layers compact their KV so memory tracks the window, not the context.

## What the planner assumes about the engine

The planner's memory model matches the engine: weights in the planned representation,
KV in f16 on GPUs and f32 on CPUs, tied embeddings duplicated at both ends of a
pipeline, ~256 MiB attention scratch, transport buffers, backend runtime overhead and a
safety margin.

## Development aids

- `TENDRIL_SIM_LATENCY_MS=3` delays every data-plane hop, to emulate a slow link
  (e.g. Wi-Fi) when all machines are one computer.
- `tendril dev-tiny-model DIR --layers N --hidden H` writes a random-weight model with a
  byte-level tokenizer for exercising the whole stack without downloads.
