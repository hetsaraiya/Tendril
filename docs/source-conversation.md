# Distributed Inference Pool Design

Source: https://chatgpt.com/share/6ab3f3f3-1efc-83e9-acba-bb50eccb5708

Main user/assistant messages in chronological order. Hidden system messages, tool calls, and reasoning omitted.

---

## User

so i have a project idea....



Let's say I'm thinking at a very, very small scale right now. I have two PCs: one is M4 and one is M5, each with 16 GB of RAM, and I want to load a model that is 19 GB of weights. It is Gemma 4.  Now, the actual project idea is to make a pool of different compute and different devices so that we can connect them and just use them bit by bit for this inference thing. Let's say this inference engine that we are making should intelligently decide what model we can run in this given pool. Let's say all the machines in this pool are M4 or M5, or maybe M-series chips, then we should decide that we should run MLX models.  Let's say a few are pure server-grade GPUs, and some of them are just simple laptops and just simple user/consumer-grade PCs. They should be used for storing all the KV cache because they will have SSDs, which can be used to store KV cache and all the things. Server-grade GPUs can be used for the computation and token prediction and all the things like this, so I want to make this product happen. What are your thoughts on this?  

2. Let's say I'm thinking at a small scale, and I'm thinking about Gemma, which is 19 GB of weights, and for M4 and M5, how can we do that?
3. How will we determine which weights are where and all the computation things?

---

## Assistant

Your core idea is feasible, but there are two very different problems hiding inside it: **splitting one model across machines** and **using heterogeneous machines as a shared inference fabric**. I’m grounding this against current distributed-inference projects first, because the design choices differ a lot depending on which one you want to solve first.

---

## Assistant

One important correction to the product idea: **SSD machines should not be treated as the normal home for the active KV cache.** Active decoding touches KV continuously, so putting it across a network on somebody else’s SSD can destroy token latency. SSD/network tiers are excellent for *cold/inactive KV, reusable prefix caches, paused sessions, and spillover*; hot KV should stay in accelerator/unified/host memory. vLLM’s current tiered design follows exactly this hierarchy: accelerator → CPU memory → filesystem/network/object-store tiers.

---

## Assistant

I also found a project you should study before writing the scheduler: **exo is already very close to the first half of your idea**. Its current design does automatic discovery, topology-aware model placement, pipeline/tensor sharding, and MLX-based distributed execution. 

That doesn’t kill your idea. The interesting extension is making the cluster a **resource fabric**, where nodes can have different roles—compute, hot-memory, cold-KV/prefix storage—and where the scheduler chooses not only *where* a model runs, but *which runtime and parallelization strategy* to use.

---

## Assistant

Yes — this is feasible, and the small two-Mac version is exactly where I would start.

The important thing I found while checking the current ecosystem is that **the basic “pool machines and shard a model across them” idea already exists** in projects such as exo and llama.cpp RPC. Exo now does topology-aware placement, pipeline/tensor parallelism, automatic discovery, and MLX distributed. llama.cpp RPC can already distribute weights and KV across local/remote devices according to available memory. citeturn889727search0turn257887search6

So the interesting product is one level higher:

> **A heterogeneous inference operating system that discovers resources and decides what each machine should do.**

That means compute placement + model format/runtime selection + memory placement + KV hierarchy + networking strategy.

### 1. What I think your architecture should become

Think of every machine joining the pool as advertising a capability profile:

| Resource | Example |
|---|---|
| Accelerator | Apple Metal / NVIDIA CUDA / CPU |
| RAM/VRAM | 16 GB unified / 80 GB HBM / 64 GB RAM |
| Runtime | MLX / CUDA / CPU |
| Compute benchmark | measured layer/sec, not just theoretical FLOPS |
| Network | Wi-Fi / 10GbE / TB4 / TB5 RDMA |
| Network latency | measured |
| SSD | capacity + read/write bandwidth |
| Current load | free RAM, GPU utilisation, temperature etc. |

Then your scheduler gets a model and produces a **placement plan**.

For example:

```text
MODEL: Gemma-4-31B-Q4
Weights: 19 GB

                  Inference Planner
                         │
              ┌──────────┴──────────┐
              │                     │
        Mac M4 - 16 GB         Mac M5 - 16 GB
        MLX / Metal            MLX / Metal
              │                     │
        embedding               layers 29-59
        layers 0-28             final_norm
              │                 lm_head
              └──── activations ───►
```

The scheduler should not start by saying “two machines = 50/50.” It should inspect the actual tensors and benchmark the machines.

---

## 2. Your exact 2 × 16 GB Mac experiment

Gemma 4 31B is especially interesting here.

Google lists approximately **17.5 GB for Gemma 4 31B Q4**, and the model has **60 transformer layers** with hidden size 5,376. citeturn187615search2turn707037view0

So your ~19 GB MLX checkpoint is in the right territory.

You theoretically have:

```text
Mac M4     16 GB
Mac M5     16 GB
----------------
Total      32 GB
```

But you cannot treat all 32 GB as model-weight memory. You still need memory for macOS, MLX/Metal, activations, KV cache, tokenizer/runtime state, networking buffers, etc.

So I would target something roughly like:

```text
Mac M4
~9 GB weights
+ KV for its layers
+ activations/runtime

Mac M5
~10 GB weights
+ KV for its layers
+ activations/runtime
```

Not literally 9.5/9.5 unless the tensor boundaries happen to produce that.

### Pipeline parallelism is what I would use first

For every token:

```text
token
  │
  ▼
Mac M4
Embedding
Layer 0
Layer 1
...
Layer 28
  │
  │ hidden state
  ▼
Mac M5
Layer 29
...
Layer 59
Final Norm
LM Head
  │
  ▼
next token
  │
  └──────────────► Mac M4
```

And repeat.

The important thing is that **you do not send weights across the network during inference**.

They stay resident:

```text
M4 RAM                 M5 RAM

W0                       W29
W1                       W30
W2                       W31
...                      ...
W28                      W59
```

Only activations cross the network.

For Gemma 4 31B, the hidden state is 5,376 elements. citeturn707037view0

At BF16 that's approximately:

```text
5376 × 2 bytes
≈ 10.5 KB
```

per generated token at your partition boundary.

That's tiny.

For an 8K-token prefill:

```text
10.5 KB × 8192
≈ 84 MB
```

So prefill cares more about **bandwidth**, while autoregressive decoding starts caring heavily about **network latency**.

This is precisely why pipeline parallelism is a very reasonable first implementation.

---

## 3. Where does the KV cache live?

This becomes surprisingly elegant with pipeline parallelism.

Mac M4 owns:

```text
layers 0–28
```

Therefore it stores:

```text
K0,V0
K1,V1
...
K28,V28
```

Mac M5 owns:

```text
layers 29–59
```

and therefore stores:

```text
K29,V29
...
K59,V59
```

You **don't need one central KV-cache machine**.

Each compute node naturally owns the KV corresponding to the attention layers it computes.

Then your larger product can introduce tiers:

```text
               HOT
                │
        Accelerator memory
                │
             RAM
                │
        Remote RAM pool
                │
          Local NVMe
                │
          Remote NVMe
                │
               COLD
```

This part of your idea is very current: vLLM now has a multi-tier KV offloading architecture where GPU/accelerator cache spills to CPU memory and then secondary tiers such as filesystem, network/P2P, or object storage. citeturn257887search0turn257887search7

So I would use random laptop SSDs for **cold KV/prefix cache**, not active token decoding.

---

# How do we determine which weights go where?

This is probably the most interesting component of your project.

Don't reason in terms of “19GB / 2.”

Your model loader should create something like:

```text
ModelManifest
│
├── embedding.weight         700 MB
│
├── layer.0                  295 MB
│   ├── attention
│   └── mlp
│
├── layer.1                  295 MB
│
...
│
├── layer.59                 295 MB
│
├── final_norm                ...
└── lm_head                   ...
```

The numbers there are illustrative—the planner should calculate actual serialized tensor sizes.

Then your scheduler solves:

```text
MINIMIZE

max(stage_compute_time)
+ network_transfer_time
+ synchronization_latency

SUBJECT TO

node_weights
+ node_KV
+ activation_memory
+ runtime_memory
< available_memory
```

So perhaps benchmark results tell you:

```text
M4 layer time = 4.2 ms
M5 layer time = 3.5 ms
```

The M5 is faster.

Instead of:

```text
30 layers │ 30 layers
```

the optimal split might become:

```text
M4                    M5

layers 0–26           layers 27–59
27 layers             33 layers
```

because now both stages take approximately the same amount of time.

That is the beginning of your **placement optimizer**.

---

## Pipeline parallel vs tensor parallel

This distinction will matter a lot.

Pipeline parallel:

```text
M4: layers 0 ─────── 28
                       │
                       ▼
M5:                 layers 29 ─────── 59
```

Tensor parallel:

```text
                 layer 17

             weight matrix W
             ┌─────┴─────┐
             │           │
           M4 half     M5 half
             │           │
             └── reduce ─┘
```

Tensor parallel means both machines participate in essentially every transformer layer, requiring collectives constantly.

MLX now officially supports tensor-parallel distributed layers, but its documentation specifically points to the low-latency **JACCL/RDMA-over-Thunderbolt-5** backend for this sort of workload. citeturn889727search2turn381738search0

Therefore on your machines I would initially choose:

**TB4/Wi-Fi/Ethernet → pipeline parallel**

**TB5 + RDMA → investigate tensor parallel**

For reference, Apple's current base M5 and M5 Air machines have Thunderbolt 4, while M5 Pro/Max have Thunderbolt 5. citeturn108172search0turn108172search1

MLX's normal ring backend works across TCP and can run across multiple Macs, including over Ethernet/Thunderbolt networking. citeturn177901search0

---

# Where your project becomes much more interesting

Eventually I would make the scheduler think in terms of **compute islands**, rather than individual machines.

Imagine:

```text
POOL
│
├── Apple Island
│   ├── M4 16GB
│   ├── M5 16GB
│   └── M4 Max 64GB
│
├── CUDA Island
│   ├── RTX 4090 24GB
│   └── RTX 4090 24GB
│
├── Memory Nodes
│   ├── PC 64GB RAM
│   └── PC 128GB RAM
│
└── Storage Nodes
    ├── Laptop 2TB NVMe
    └── NAS 8TB
```

User asks:

```text
run Gemma-4-31B
```

Your planner could respond internally:

```text
Requirements:
  weight_memory = 19GB
  estimated_KV = X
  target_context = 32K

Candidates:

Apple:
  MLX
  pipeline parallel
  M4 + M5
  predicted = 9 tok/s

CUDA:
  MLX-CUDA / vLLM / other adapter
  single 4090
  predicted = 16 tok/s

Selected:
  CUDA node

KV:
  hot -> GPU
  warm -> local RAM
  cold prefix -> PC-Storage-03
```

THAT is a considerably more interesting system than simply distributed inference.

---

## One thing I would deliberately **not** attempt in v1

Don't immediately try:

```text
M4 Metal
   +
RTX 4090 CUDA
   +
AMD GPU ROCm
```

inside the same tensor-parallel forward pass.

Different runtimes, kernels, quantization layouts and communication primitives make that extremely ugly.

Instead:

```text
compatible devices
        ↓
compute group / island
        ↓
one model execution
```

Other heterogeneous machines can provide storage, cache, routing, or run other model stages.

Cross-runtime pipeline execution can come much later.

---

## The first version I would build

I would make v0 almost absurdly constrained:

```text
2 Apple Silicon Macs
        │
        ├── auto-discover each other
        ├── benchmark RAM/network/MLX
        ├── inspect an MLX Gemma checkpoint
        ├── automatically divide layers
        ├── load only assigned weights
        ├── distributed KV ownership
        └── expose one OpenAI-compatible endpoint
```

The really important part is this:

```text
./cluster start

Nodes discovered:
  M4  16GB
  M5  16GB

Model requested:
  gemma-4-31b-q4
  19.0GB

Planner:
  Runtime: MLX
  Strategy: Pipeline Parallel
  Node M4: layers 0-26
  Node M5: layers 27-59
  Predicted RAM: 12.1GB / 13.0GB
  Link: Thunderbolt TCP
  Status: feasible

Loading...
```

No hand-written `--tensor-split`.

No user deciding layers.

**The planner decides.**

That is the part of this project I think is genuinely worth pursuing.

And I'd treat **exo, llama.cpp RPC, MLX Distributed, and vLLM's new tiered KV system as references/components to beat or combine—not things to unknowingly rebuild.** Exo in particular is close enough that studying its current placement implementation should be one of the first research steps. citeturn597560search1turn257887search6

---

## User

How would we plan the optimal split of weights if it's not MoE and it's a dense model?

---

## Assistant

For a **dense transformer**, the problem is actually cleaner than MoE because **every token uses essentially every transformer layer**. There is no expert routing to optimize.

For your first version, I would treat the model as an ordered graph:

```text
Embedding
   ↓
Layer 0
   ↓
Layer 1
   ↓
...
   ↓
Layer 59
   ↓
Norm
   ↓
LM Head
```

and solve **where to cut that graph**.

### Example: two Macs

Suppose profiling the actual checkpoint gives us:

```text
Gemma dense model
Weights: 19 GB

Embedding       0.7 GB
Layers 0-59    17.7 GB
Norm/head       0.6 GB
```

Numbers are illustrative.

We benchmark both machines first:

```text
             M4                 M5
RAM free     13.0 GB            13.2 GB
layer speed  5.0 ms/layer       3.7 ms/layer
```

A naive memory-only scheduler would do:

```text
M4: layers 0-29
M5: layers 30-59
```

But that isn't optimal because M5 is faster.

Instead, our planner searches possible cut points:

```text
cut after layer 20
cut after layer 21
cut after layer 22
...
cut after layer 40
```

For every cut `k`, calculate:

\[
T(k)=T_{M4}(0..k)+T_{network}+T_{M5}(k+1..59)
\]

subject to:

\[
Weights_i + KV_i + Activations_i + RuntimeOverhead_i < AvailableMemory_i
\]

For example, it might discover:

```text
             M4                  M5
Layers       0-25                26-59
Count        26                  34

Weight       ~8 GB               ~11 GB
Compute      130 ms              126 ms
```

That's much better balanced.

---

### But don't optimize based on GB alone

Your planner should profile **every transformer block**.

Something like:

```json
{
  "layer": 17,
  "weight_bytes": 302871552,
  "kv_bytes_per_token": 65536,
  "m4_decode_us": 4870,
  "m5_decode_us": 3560,
  "activation_bytes": 10752
}
```

Then build a matrix:

```text
             M4       M5
Layer 0      4.8ms    3.5ms
Layer 1      4.7ms    3.5ms
Layer 2      4.9ms    3.6ms
...
Layer 59     4.6ms    3.4ms
```

Now placement becomes an optimization problem rather than a heuristic.

Exo is moving in this direction: its current topology-aware placement accounts for device resources plus network latency/bandwidth rather than simply counting machines. citeturn394547search2

---

## There are actually two completely different ways to split a dense model

### A. Layer / pipeline partitioning

This is what I'd build first.

```text
M4                         M5

Layer 0
Layer 1
...
Layer 25
    │
    │ activation
    └──────────────────────► Layer 26
                              Layer 27
                              ...
                              Layer 59
```

Weights are completely independent:

```text
M4 stores W0...W25
M5 stores W26...W59
```

You only communicate at the boundary.

This gives you **very little network communication**, which makes it suitable for ordinary Macs connected over Ethernet/Thunderbolt.

There is one important caveat:

> For a single autoregressive request, pipeline splitting mainly lets the model **fit**; it doesn't magically halve token latency.

One token still has to go:

```text
M4 → M5 → token
M4 → M5 → token
M4 → M5 → token
```

You get much better throughput when you have multiple sequences/microbatches moving through the pipeline simultaneously.

---

### B. Tensor parallelism

Instead of assigning whole layers:

```text
             Transformer Layer 17

          ┌──────────┴──────────┐
          │                     │
         M4                    M5
     half of Wq             half of Wq
     half of Wk             half of Wk
     half of Wv             half of Wv
     half of FFN            half of FFN
          │                     │
          └──── collective ─────┘
```

Both Macs participate in **every layer**.

MLX supports exactly this kind of dense-layer sharding. Its distributed layers can shard linear matrices across devices by input/output dimension and then perform distributed reductions where necessary. citeturn394547search0

This can provide actual compute acceleration, but you communicate **inside every transformer block**.

So:

```text
Pipeline parallel
communication: ~once per stage
network demand: low
heterogeneous hardware: easier

Tensor parallel
communication: multiple times per layer
network demand: very high
heterogeneous hardware: harder
```

---

## Your scheduler should therefore make two decisions

Not:

```text
Where do I put these 19 GB?
```

but:

```text
1. WHICH PARALLELISM STRATEGY?
2. GIVEN THAT STRATEGY, WHICH WEIGHTS GO WHERE?
```

So maybe your planner evaluates:

```text
Model: Gemma
19 GB

Nodes:
 M4 16GB
 M5 16GB

Network:
 1Gb Ethernet
 RTT: 0.4ms

Candidate A:
 Pipeline
 cut = layer 26
 predicted decode = 7.8 tok/s

Candidate B:
 Tensor Parallel 2-way
 communication overhead = huge
 predicted decode = 3.1 tok/s

SELECT: Pipeline
```

But connect two powerful Macs through very high-bandwidth, very-low-latency RDMA:

```text
Candidate A:
 Pipeline = 8 tok/s

Candidate B:
 TP=2 = 13 tok/s

SELECT: TP
```

That's why MLX's TP implementation and Exo's topology-aware approach matter: network topology becomes part of model placement. citeturn394547search0turn394547search2

---

## How I would implement the actual optimizer

For pipeline parallelism, this is essentially a **graph partitioning / constrained dynamic-programming problem**.

Given:

```text
L = transformer layers
N = devices

weight[layer]
kv[layer][context]
compute[layer][device]

bandwidth[A][B]
latency[A][B]
available_memory[device]
```

Find:

```text
partition(model, devices)
```

that minimizes something such as:

\[
\text{Cost} =
\alpha(\text{decode latency})
+
\beta(\text{prefill latency})
+
\gamma(\text{network cost})
+
\delta(\text{memory pressure})
\]

while never exceeding node memory.

For two machines, you can literally brute-force **every valid boundary**. With 60 layers that's only ~59 meaningful cuts.

```python
best = None

for cut in range(1, 60):

    left = layers[:cut]
    right = layers[cut:]

    if not fits(left, M4):
        continue

    if not fits(right, M5):
        continue

    score = (
        decode_cost(left, M4)
        + network_cost(M4, M5)
        + decode_cost(right, M5)
    )

    if best is None or score < best.score:
        best = Placement(cut, score)
```

For 10–100 heterogeneous machines it becomes much more interesting and you'd move toward DP, graph partitioning or an optimizer.

### And one more important improvement

Don't force the partition boundary to be **between transformer blocks forever**.

Eventually your planner could decide:

```text
M4:
  Layers 0-24

Layer 25:
  tensor-parallel M4 + M5

M5:
  Layers 26-59
```

or create groups:

```text
M4 + M5         = MLX TP group
4090 + 4090     = CUDA TP group

             ↓

pipeline between groups
```

That gives you **hierarchical parallelism**.

I think that is eventually the heart of your product:

> **Profile the model graph + profile the compute/network topology → automatically construct the cheapest feasible distributed execution graph.**

Dense models don't make this less interesting—they actually give you a much cleaner foundation on which to build the planner.

---

## User

what should i name this project give me somehting very creative and product nature realated

---

## Assistant

I’d lean into **nature + swarm + distributed intelligence** rather than “cluster/inference” sounding names.

- **Mycel** — from *mycelium*, a distributed underground network. Very fitting.
- **Canopy** — many nodes forming one shared compute layer.
- **Rhizome** — decentralized root network; technically on-theme.
- **Spore** — machines joining and spreading compute capacity. Strong product name.
- **Murmur** — like murmuration; many independent nodes behaving as one.
- **Lichen** — symbiotic distributed system made of different organisms.
- **Tendril** — compute/resources extending between devices.
- **Grove** — individual machines = trees, cluster = grove.
- **Kelp** — distributed network, flexible and interconnected.
- **Nectar** — resources gathered from many places into one usable pool.
- **Synapse** — less nature-heavy, but fits devices coordinating dynamically.
- **Meshling** — playful product-ish name around a compute mesh.

My strongest 5 for this project are **Mycel**, **Spore**, **Rhizome**, **Canopy**, and **Murmur**.

If I had to pick **one**: **Mycel**. It sounds like an actual infra product, and the metaphor is almost perfect: many independent nodes underneath acting like one larger organism.

---

## User

combine things with greek gods

---

## Assistant

Yes — Greek mythology + nature + distributed compute can give much stronger product names.

- **GaiaMesh** — Gaia = Earth; perfect for a living network of machines.
- **HermesGrid** — Hermes = speed + communication; great for routing compute between nodes.
- **AtlasMesh** — Atlas carries the world; fits shared distributed compute.
- **Prometheus Grid** — Prometheus brought fire/technology; strong infra vibe.
- **NyxMesh** — Nyx = night; darker, premium infra-product feel.
- **AetherGrid** — Aether = upper sky / pure air; sounds very cloud-infra.
- **GaiaForge** — distributed machines collectively forging inference.
- **Hermes Grove** — fast communication across a “grove” of machines.
- **Atlas Canopy** — many devices forming one compute canopy.
- **AetherRoot** — invisible infrastructure connecting machines underneath.
- **Prometheus Grove** — collective compute with a nature metaphor.
- **GaiaSpore** — nodes spreading/joining like spores.
- **Atlas Mycel** — mycelium network + Atlas-scale compute.
- **AetherBloom** — compute capacity “blooms” as machines join.
- **Hermes Mycel** — distributed network optimized around fast communication.

The ones that sound most like **real infra products** to me are:

**Aether**, **GaiaMesh**, **AtlasMesh**, **AetherGrid**, **GaiaForge**, and **HermesGrid**.

For your specific project, I especially like **AetherMesh**.

> **AetherMesh — heterogeneous compute, acting as one machine.**

That one feels very product/company-name ready.

---

## User

give me a whole prompt to make this shit in rust.. make sure that covers all the edge cases nad all the scenarios

---

## Assistant

I’m going to turn this into a **coding-agent master prompt**, not a vague PRD: architecture, Rust crate layout, scheduling model, node discovery, model sharding, transport, KV tiers, failure handling, benchmarking, observability, security, and staged milestones. I’m also checking the current distributed-inference/runtime landscape so the prompt doesn’t bake in dead assumptions.

---

## Assistant

The biggest implementation decision I’m locking into the prompt is: **Rust owns the entire control plane, scheduler, networking, cache manager, API, and worker lifecycle; accelerator execution is behind a Rust `RuntimeBackend` trait.** On Apple Silicon, use MLX through Rust/`mlx-c` rather than inventing Metal kernels in v1. Rust MLX bindings exist today, but several are still explicitly “active development,” so the adapter boundary is important. 

For networking, I’d separate a small control plane from the activation/KV data plane. QUIC is a good first data transport because Rust’s Quinn gives multiplexed streams plus TLS 1.3, while leaving room for a later RDMA transport. Current MLX and llama.cpp both show why that matters: TCP/ring works generally, while low-latency tensor parallelism benefits heavily from RDMA-class links.

---

## Assistant

I’d use this as the **master build prompt** for Cursor/Codex/Claude Code. I’m assuming the working name **AetherMesh**; replace it later if needed.

:::writing{variant="document" id="48317" title="AetherMesh — Master Engineering Prompt"}
# AetherMesh

You are the principal distributed-systems + inference-runtime engineer responsible for designing and implementing **AetherMesh**, a distributed heterogeneous LLM inference fabric written primarily in Rust.

Do not build a toy demo. Build the foundation of a real product.

The system must discover available computers, understand their hardware/network/storage capabilities, inspect a requested model, determine whether the model can run on the available cluster, automatically create an execution plan, distribute model weights, execute distributed inference, manage KV cache placement, tolerate failures, and expose a simple OpenAI-compatible inference API.

The long-term goal is:

> Connect arbitrary compute devices into a pool and automatically turn them into the best possible inference cluster for a requested model.

Examples of nodes:

- Apple Silicon MacBooks/Mac minis
- NVIDIA GPU servers
- consumer desktops
- CPU-only machines
- high-RAM machines
- storage-heavy machines
- heterogeneous combinations of the above

However, **do not attempt everything at once**.

The first production milestone is:

> Two Apple Silicon machines, each with 16 GB unified memory, automatically running a dense MLX-format model whose weights cannot fit comfortably on either machine individually.

Example target:

```text
Mac M4 16 GB
Mac M5 16 GB

        ↓

AetherMesh cluster

        ↓

~19 GB dense model

        ↓

automatic model partition

M4:
  embedding
  layers 0..K

M5:
  layers K+1..N
  final norm
  LM head

        ↓

distributed autoregressive inference
```

The user must NOT manually specify the layer split.

AetherMesh determines it.

---

# 1. NON-NEGOTIABLE DESIGN PRINCIPLES

Follow these rules throughout the implementation.

## 1.1 Rust owns the product

The following must be implemented in Rust:

- CLI
- daemon
- discovery
- cluster membership
- hardware profiling
- network profiling
- model inspection
- model manifests
- scheduler
- placement optimizer
- weight distribution
- distributed execution orchestration
- transport
- KV cache coordination
- failure detection
- API server
- observability
- persistence
- security
- tests
- simulator

Do not introduce Python services into the production architecture.

Accelerator libraries may internally contain C/C++/Metal/CUDA code and may be accessed through Rust FFI.

That is acceptable.

---

# 2. RUNTIME ABSTRACTION

Never couple the scheduler directly to MLX.

Create:

```rust
trait RuntimeBackend {
    fn runtime_kind(&self) -> RuntimeKind;

    async fn inspect_model(
        &self,
        model: &ModelSource,
    ) -> Result<ModelManifest>;

    async fn validate_plan(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<PlanValidation>;

    async fn load_partition(
        &self,
        partition: ModelPartition,
    ) -> Result<LoadedPartition>;

    async fn prefill(
        &self,
        request: PrefillRequest,
    ) -> Result<StageOutput>;

    async fn decode(
        &self,
        request: DecodeRequest,
    ) -> Result<StageOutput>;

    async fn unload(
        &self,
        model_id: ModelId,
    ) -> Result<()>;
}
```

Initial backend:

```text
MLX
```

Future:

```text
MLX
CUDA
ROCm
CPU/GGML
Metal-native
remote accelerator
```

Runtime capabilities must be explicit.

Example:

```rust
struct RuntimeCapabilities {
    supports_pipeline_parallel: bool,
    supports_tensor_parallel: bool,
    supports_kv_quantization: bool,
    supports_weight_quantization: Vec<Quantization>,
    supported_dtypes: Vec<DType>,
    supported_model_architectures: Vec<ModelArchitecture>,
    max_tensor_bytes: Option<u64>,
}
```

Never assume a runtime can execute a model merely because weights fit into memory.

---

# 3. APPLE SILICON RUNTIME

For v1, execute MLX models from Rust.

Prefer an existing maintained Rust binding over `mlx-c` where it exposes everything required.

If a binding cannot support the required model architecture or operation:

```text
Rust
 ↓
mlx-c FFI
 ↓
MLX
 ↓
Metal
```

Implement the missing API through a small isolated FFI crate.

Do NOT spread raw FFI throughout the codebase.

Structure it as:

```text
runtime-mlx
    safe Rust API
        ↓
runtime-mlx-sys
        ↓
mlx-c
```

Never pretend unsupported functionality works.

Feature-gate experimental capabilities.

---

# 4. ARCHITECTURE

Use a workspace structured approximately like:

```text
aethermesh/
├── Cargo.toml
│
├── crates/
│   ├── aether-core/
│   ├── aether-protocol/
│   ├── aether-discovery/
│   ├── aether-hardware/
│   ├── aether-topology/
│   ├── aether-model/
│   ├── aether-planner/
│   ├── aether-runtime/
│   ├── aether-runtime-mlx/
│   ├── aether-transport/
│   ├── aether-kv/
│   ├── aether-storage/
│   ├── aether-scheduler/
│   ├── aether-observability/
│   └── aether-simulator/
│
├── bins/
│   ├── aether/
│   └── aetherd/
│
├── proto/
├── tests/
├── benchmarks/
└── docs/
```

Keep crates cohesive.

Do not create dozens of meaningless micro-crates.

---

# 5. PROCESS MODEL

Every physical machine runs:

```text
aetherd
```

The daemon provides:

```text
Hardware Agent
Runtime Worker
Model Cache
KV Manager
Network Agent
Health Reporter
Execution Worker
```

One node acts as the cluster coordinator for v1.

The coordinator manages:

```text
membership
topology
scheduling
model placement
request routing
plan epochs
failure handling
```

Coordinator failure must be detectable.

Full distributed consensus is NOT required for v1.

Design interfaces so Raft/consensus can be introduced later.

---

# 6. CLI

Required commands:

```bash
aether init
aether join <address>
aether leave

aether nodes
aether topology

aether model inspect <model>
aether model pull <model>
aether model remove <model>

aether plan <model>
aether plan <model> --context 32768

aether serve <model>

aether status
aether doctor
aether benchmark
```

Example UX:

```text
$ aether nodes

NODE       CHIP       RAM     FREE     RUNTIME   LINK
m4-air     M4         16GB    12.7GB   MLX       8.2Gbps / 0.31ms
m5-air     M5         16GB    13.1GB   MLX       8.6Gbps / 0.28ms
```

Then:

```text
$ aether plan gemma-model
```

Example result:

```text
Model
  Architecture: Gemma
  Type: Dense
  Weight size: 19.1 GB
  Layers: 60

Cluster
  Nodes: 2
  Available memory: 25.8 GB

Candidate strategies:

  Pipeline
    feasible: yes
    predicted decode: ...
    predicted prefill: ...
    network cost: ...

  Tensor Parallel
    feasible: no
    reason: transport latency exceeds threshold

Selected strategy:
  Pipeline Parallel

Placement:

  m4-air
    embedding
    layers 0-25

  m5-air
    layers 26-59
    final_norm
    lm_head

Estimated:
  M4 peak memory: ...
  M5 peak memory: ...
  Boundary traffic/token: ...
```

Explain planner decisions.

Never return merely:

```text
no feasible plan
```

Return why.

---

# 7. HARDWARE DISCOVERY

Create a normalized:

```rust
struct NodeCapabilities {
    node_id: NodeId,

    os: OperatingSystem,
    architecture: CpuArchitecture,

    cpu: CpuInfo,
    accelerators: Vec<AcceleratorInfo>,

    total_memory_bytes: u64,
    available_memory_bytes: u64,

    disks: Vec<DiskInfo>,
    runtimes: Vec<RuntimeCapabilities>,

    interfaces: Vec<NetworkInterface>,

    thermal: Option<ThermalState>,
    power: Option<PowerState>,
}
```

On Apple Silicon determine at minimum:

```text
chip family
CPU cores
GPU cores where available
total unified memory
currently available memory
macOS version
Metal availability
MLX availability/version
disk capacity
disk free space
disk benchmark
network interfaces
Thunderbolt availability
Ethernet
Wi-Fi
```

Do not schedule based only on advertised hardware.

Benchmark it.

---

# 8. ACTIVE PROFILING

Each node must maintain measured performance.

Examples:

```rust
struct NodeBenchmark {
    memory_bandwidth: f64,
    disk_read_bandwidth: f64,
    disk_write_bandwidth: f64,

    matmul_score: f64,

    model_layer_profiles: HashMap<ModelProfileKey, LayerProfile>,

    measured_at: SystemTime,
}
```

Measurements must expire.

Old measurements cannot be trusted forever.

Account for:

```text
thermal throttling
other applications consuming memory
battery operation
current GPU load
network congestion
```

Re-profile selectively instead of benchmarking the entire machine before every request.

---

# 9. NETWORK TOPOLOGY

The cluster is NOT a flat list of machines.

Represent it as a graph.

```rust
struct NetworkEdge {
    source: NodeId,
    destination: NodeId,

    bandwidth_bytes_per_sec: f64,

    latency_p50: Duration,
    latency_p95: Duration,
    jitter: Duration,
    packet_loss: f64,

    transport: TransportKind,
}
```

Measure connections independently.

Do not assume:

```text
A -> B == B -> A
```

Test:

```text
small-message latency
medium transfer
large sequential transfer
concurrent streams
```

Topology information must affect scheduling.

---

# 10. DISCOVERY

Local cluster discovery should be zero-config where possible.

Support:

```text
local automatic discovery
manual peer address
static configuration
```

Nodes must possess persistent identities.

Do not trust a node merely because it broadcasts on the LAN.

Each node has:

```text
NodeId
public key
private key
```

Joining requires cluster authorization.

---

# 11. CONTROL PLANE VS DATA PLANE

Separate them.

## Control plane

Use small structured RPC messages.

Suitable operations:

```text
JoinCluster
Heartbeat
Capabilities
BenchmarkReport
LoadModel
UnloadModel
InstallPlan
StartRequest
CancelRequest
KVLocationUpdate
NodeDrain
Health
```

A Rust gRPC implementation such as tonic is appropriate.

## Data plane

Large or latency-sensitive data includes:

```text
activations
prefill tensors
KV blocks
model chunks
```

Use a transport abstraction:

```rust
trait DataTransport {
    async fn send_tensor(...);
    async fn recv_tensor(...);
    async fn send_blob(...);
}
```

Initial implementation:

```text
QUIC
```

Later implementations:

```text
TCP
Thunderbolt RDMA
RoCE
InfiniBand
shared memory
NCCL
JACCL
```

Do not bake QUIC into execution logic.

---

# 12. BACKPRESSURE

Never allow unbounded network queues.

Every stream must support:

```text
bounded buffers
flow control
timeouts
cancellation
maximum tensor sizes
maximum outstanding bytes
```

Slow consumers must propagate pressure upstream.

Do not allow one slow worker to OOM another node through queued activations.

---

# 13. MODEL MANIFEST

Never schedule using only total checkpoint size.

Inspect the actual model.

Produce:

```rust
struct ModelManifest {
    model_id: ModelId,
    architecture: ModelArchitecture,

    model_kind: ModelKind,

    dtype: DType,
    quantization: Option<Quantization>,

    hidden_size: usize,
    num_layers: usize,

    vocab_size: usize,

    components: Vec<ModelComponent>,

    total_weight_bytes: u64,

    kv_layout: KVLayout,

    tied_weights: Vec<TiedWeightGroup>,
}
```

`ModelKind`:

```rust
enum ModelKind {
    Dense,
    MixtureOfExperts,
    Multimodal,
    Unknown,
}
```

For this milestone support:

```text
Dense
```

Reject unsupported architectures clearly.

---

# 14. MODEL COMPONENT GRAPH

Represent components explicitly:

```text
embedding
layer 0
layer 1
...
layer N
final norm
lm head
```

Each component includes:

```rust
struct ModelComponent {
    id: ComponentId,

    kind: ComponentKind,

    weight_bytes: u64,

    input_shape: TensorShape,
    output_shape: TensorShape,

    dependencies: Vec<ComponentId>,

    estimated_scratch_bytes: u64,
}
```

Do not assume all transformer layers have equal cost.

Some architectures can contain:

```text
global attention
local attention
sliding window attention
different KV dimensions
special layers
shared weights
different MLP sizes
```

Inspect rather than guess.

---

# 15. SAFETENSORS

Parse SafeTensors metadata directly from Rust.

Support sharded checkpoints:

```text
model-00001-of-N.safetensors
...
model-N-of-N.safetensors
model.safetensors.index.json
```

The planner must know:

```text
tensor -> file
tensor -> byte range
tensor -> layer
tensor -> dtype
tensor -> shape
```

Avoid reading entire checkpoints merely to inspect metadata.

Use memory mapping where safe and appropriate.

---

# 16. WEIGHT STORAGE

Weights must be content-addressed.

Each file/chunk must have:

```text
hash
size
model identity
revision
```

Support:

```text
resume download
partial download
checksum validation
local cache
peer-to-peer model distribution
atomic commit
```

Never expose a partially downloaded model as ready.

Use:

```text
.tmp
```

then rename atomically after validation.

---

# 17. DENSE MODEL PLANNING

Dense models execute every transformer layer for every token.

For v1, implement contiguous pipeline partitioning.

Example:

```text
Node A
layers 0..23

       activation

Node B
layers 24..59
```

Never distribute random individual tensors merely to achieve equal byte counts.

Primary unit:

```text
whole transformer stage
```

Special components such as embedding/head may be separate planner units.

---

# 18. MEMORY MODEL

A placement is feasible only when:

```text
weights
+ KV cache
+ activations
+ temporary/scratch buffers
+ runtime allocations
+ transport buffers
+ allocator fragmentation allowance
+ operating system reserve
+ safety margin

<= usable memory
```

Never use:

```text
weight_size < free_RAM
```

as the feasibility test.

Define:

```rust
struct MemoryEstimate {
    weight_bytes: u64,
    kv_bytes: u64,
    activation_bytes: u64,
    scratch_bytes: u64,
    transport_bytes: u64,
    runtime_reserve_bytes: u64,
    safety_margin_bytes: u64,

    peak_bytes: u64,
}
```

Default safety margin should be configurable.

Memory calculations must be overflow-safe.

Use checked arithmetic.

---

# 19. KV MEMORY

KV size depends on:

```text
layers owned
KV heads
head dimension
dtype
sequence length
batch size
number of active requests
KV quantization
```

Never estimate KV solely from parameter count.

Planner input includes:

```text
max context
expected concurrency
batching configuration
KV dtype
```

Example:

```bash
aether plan model \
  --context 32768 \
  --concurrency 4
```

A model might fit at:

```text
4K context
```

but not:

```text
128K context
```

The planner must report that distinction.

---

# 20. KV OWNERSHIP

Pipeline stage owns KV corresponding to layers it executes.

Example:

```text
Node A
layers 0-20
KV 0-20

Node B
layers 21-59
KV 21-59
```

Avoid remote access to hot KV during normal decode.

---

# 21. KV CACHE TIERS

Implement a generic tier model:

```text
T0 accelerator/unified memory
T1 local system RAM
T2 local NVMe
T3 remote RAM
T4 remote NVMe/object storage
```

Hot decode KV belongs as close to compute as possible.

Do NOT place active per-token KV on remote SSD merely because SSD capacity exists.

Secondary tiers are primarily for:

```text
prefix caching
inactive sessions
paused requests
cold KV
overflow
migration
recovery
```

Promotion should look like:

```text
SSD
 ↓
RAM
 ↓
accelerator/unified memory
```

and not:

```text
SSD -> accelerator on every token
```

---

# 22. KV BLOCKING

Do not manage KV as one gigantic byte array.

Use pages/blocks.

Example:

```rust
struct KVBlockId {
    model: ModelId,
    layer_group: LayerGroup,
    sequence: SequenceId,
    block_index: u64,
}
```

Support:

```text
reference counting
LRU initially
checksums
async writes
async promotion
eviction
pinning
```

Never evict a block while it is being consumed.

---

# 23. PREFIX CACHE

Design prefix caching even if not fully implemented in milestone 1.

Hash cache keys from:

```text
model revision
token IDs
adapter/LoRA identity
sampling-independent model state
relevant runtime configuration
```

Do not use prompt text alone as the cache identity.

---

# 24. PIPELINE EXECUTION

For each generated token:

```text
token
 ↓
stage 0
 ↓ activation
stage 1
 ↓ activation
...
 ↓
final stage
 ↓
logits
 ↓
sampling
 ↓
next token
```

Request state contains:

```rust
struct InferenceRequestState {
    request_id: RequestId,
    plan_epoch: PlanEpoch,
    model_id: ModelId,

    sequence_id: SequenceId,
    token_position: usize,

    status: RequestStatus,

    cancellation: CancellationToken,
}
```

All messages must contain:

```text
request_id
sequence_id
token position
plan epoch
stage ID
```

Reject stale messages.

---

# 25. ACTIVATION TRANSFER

Tensor messages must include:

```rust
struct TensorHeader {
    request_id: RequestId,
    plan_epoch: PlanEpoch,

    source_stage: StageId,
    destination_stage: StageId,

    token_start: u64,
    token_count: u64,

    dtype: DType,
    shape: Vec<u64>,

    payload_bytes: u64,
    checksum: Option<Checksum>,
}
```

Validate:

```text
dimensions
dtype
byte count
maximum size
stage
request
epoch
```

before allocating destination buffers.

This prevents malformed peers from forcing gigantic allocations.

---

# 26. PREFILL VS DECODE

Treat these separately.

Prefill characteristics:

```text
large activation transfers
high compute utilization
bandwidth sensitive
large temporary memory
```

Decode:

```text
tiny batches
latency sensitive
KV heavy
network RTT sensitive
```

A placement may be good for prefill and bad for decode.

Planner must calculate both.

---

# 27. PLANNER OBJECTIVES

Support:

```rust
enum OptimizationGoal {
    Balanced,
    LowestLatency,
    HighestThroughput,
    LowestNetworkTraffic,
    LowestMemoryPressure,
}
```

Default:

```text
Balanced
```

Planner score should consider:

```text
feasibility
memory headroom
prefill latency
decode latency
pipeline throughput
network traffic
network RTT
node reliability
thermal state
current utilization
model load cost
```

Never combine unrelated values without normalization.

Expose the scoring breakdown.

---

# 28. TWO-NODE OPTIMAL SPLIT

For a two-node dense transformer:

Evaluate every legal layer boundary.

For:

```text
N transformer layers
```

evaluate approximately:

```text
1 .. N-1
```

plus legal placement of embedding/norm/head.

For every boundary calculate:

```text
memory_A
memory_B

prefill_compute_A
prefill_compute_B

decode_compute_A
decode_compute_B

A->B activation transfer

expected pipeline throughput

memory headroom
```

Reject infeasible cuts.

Choose lowest objective score.

Do not default to 50/50.

---

# 29. COMPUTE PROFILING

Do not assume:

```text
M5 = X% faster than M4
```

Measure it.

Maintain per-node/per-model-family estimates:

```rust
struct ComponentBenchmark {
    prefill_ns_per_token: f64,
    decode_ns: f64,
    peak_memory_bytes: u64,
}
```

A fast approximation may benchmark representative layers.

Allow progressively better measurements after actual inference.

Planner should learn from observed execution.

Use an exponential moving average or another robust estimator.

Do not let one anomalous request permanently poison profiling data.

---

# 30. N-NODE PARTITIONING

For `N` pipeline nodes, solve contiguous partitioning.

Initially implement dynamic programming.

Objective for throughput:

```text
minimize maximum stage time
```

while respecting memory.

For latency-oriented planning:

```text
minimize total compute
+ boundary communication
```

Include network topology.

---

# 31. NODE ORDERING

Device order matters.

Example:

```text
A -> B -> C
```

may be much better than:

```text
A -> C -> B
```

due to network topology.

For small node counts:

```text
enumerate candidate orders
```

For larger clusters:

```text
heuristic search
beam search
graph optimization
```

Do not attempt factorial enumeration for a 100-node cluster.

---

# 32. PIPELINE THROUGHPUT

For one sequential autoregressive request, pipeline splitting primarily enables the model to fit across devices.

It does NOT automatically divide token latency by node count.

Model this correctly.

With multiple requests/microbatches, pipeline stages may execute concurrently.

Approximate steady-state throughput using:

```text
slowest pipeline stage
```

not sum of all stage times.

Latency and throughput are different objectives.

---

# 33. TENSOR PARALLELISM

Design interfaces for tensor parallelism but do NOT make it milestone-1 critical.

Tensor parallel execution requires much more frequent synchronization.

Only consider it when:

```text
runtime supports it
devices are compatible
network latency sufficiently low
bandwidth sufficiently high
required collectives are available
```

Do not combine arbitrary slow Wi-Fi machines into a TP group.

Future plan representation:

```text
Pipeline Stage 0
    TP group:
      node A
      node B

Pipeline Stage 1
    TP group:
      node C
      node D
```

This enables hierarchical parallelism.

---

# 34. HETEROGENEOUS COMPUTE ISLANDS

Eventually group compatible nodes:

```text
Apple MLX island

CUDA island

ROCm island

CPU island
```

Do not require all nodes in the global cluster to use the same runtime.

But do not attempt arbitrary cross-runtime tensor parallelism.

Cross-runtime composition should initially happen at:

```text
pipeline boundaries
```

where tensor interchange format can be normalized.

---

# 35. MODEL LOAD PROCESS

Loading a model must be transactional.

State machine:

```text
DISCOVERED
DOWNLOADING
VERIFYING
READY
LOADING
LOADED
FAILED
EVICTING
```

Workers must not advertise `LOADED` until all required weights are available and successfully initialized.

---

# 36. PLAN VERSIONING

Execution plans are immutable.

Every plan has:

```text
PlanId
PlanEpoch
ModelRevision
Node assignments
Runtime versions
```

Changing placement creates a new epoch.

Requests already executing remain bound to their original plan unless explicitly migrated.

Never mutate a live plan underneath a request.

---

# 37. NODE HEARTBEATS

Each worker sends:

```text
health
available memory
runtime status
model status
queue depth
thermal state
network state
```

Use leases.

Do not immediately declare a node dead after one missed heartbeat.

States:

```text
HEALTHY
SUSPECT
UNREACHABLE
DRAINING
DEAD
```

Use hysteresis.

---

# 38. NODE FAILURE DURING GENERATION

Handle this explicitly.

If a pipeline node disappears:

1. stop scheduling additional tokens through the broken plan;
2. cancel pending stage work;
3. determine whether another compatible node can host the lost partition;
4. determine whether lost KV exists in another tier;
5. if KV can be restored, load partition + KV and continue;
6. otherwise recompute KV from the token history/prompt where possible;
7. if recovery is impossible, terminate request cleanly.

Never silently generate from corrupted/incomplete KV.

Expose:

```text
finish_reason = "infrastructure_error"
```

when necessary.

---

# 39. MID-TOKEN FAILURE

A token is not committed until the final sampling decision succeeds.

If a stage crashes while processing token `T`:

```text
T remains uncommitted
```

Retry from the last committed token.

Design generation state around this rule.

---

# 40. DUPLICATE MESSAGES

Networks retry.

RPCs retry.

Connections reconnect.

All important operations must be idempotent.

Examples:

```text
LoadPartition
InstallPlan
StoreKVBlock
CancelRequest
CompleteToken
```

Use IDs and sequence numbers.

---

# 41. OUT-OF-ORDER DATA

Do not assume network completion order equals generation order.

Use:

```text
request ID
sequence ID
position
stage
epoch
```

Drop stale or duplicate activation messages.

---

# 42. REQUEST CANCELLATION

If the client disconnects:

```text
cancel generation
release scheduler reservation
release temporary activations
unpin KV when appropriate
propagate cancellation to all stages
```

Do not allow zombie inference.

---

# 43. OVERLOAD

Scheduler must expose queueing.

Node queues are bounded.

Possible policy:

```text
reject
queue
route elsewhere
```

API returns an appropriate overload error instead of exhausting RAM.

---

# 44. MEMORY PRESSURE DURING EXECUTION

Free memory can change after planning.

Workers must enforce a hard runtime allocation budget.

If crossing warning threshold:

```text
evict cold KV
reduce batching
reject new requests
```

If crossing critical threshold:

```text
stop accepting work
signal scheduler
```

Do not wait for OS OOM termination.

---

# 45. MODEL CACHE EVICTION

Model cache and KV cache have separate policies.

Never delete model files currently mapped by an active worker.

Use leases/refcounts.

Eviction considers:

```text
last access
model size
load cost
remote availability
active references
```

---

# 46. DISK FAILURE

Handle:

```text
disk full
read-only disk
I/O timeout
corrupt model
corrupt KV
removed external drive
```

KV cache corruption may invalidate cache and recompute.

Weight corruption requires re-verification/redownload.

Never execute corrupt weights.

---

# 47. NETWORK PARTITION

Distinguish:

```text
worker dead
```

from:

```text
worker temporarily unreachable
```

Coordinator must not instantly schedule the same mutable session onto multiple independent owners unless consistency rules support it.

Use epochs/leases to prevent split-brain execution.

---

# 48. SECURITY

Never expose an unauthenticated remote execution service.

Requirements:

```text
authenticated peers
encrypted transport
cluster join authorization
node identity validation
request size limits
tensor size limits
rate limits
path sanitization
```

Never accept arbitrary filesystem paths from another node without validation.

Never deserialize untrusted native objects.

Prefer explicit protobuf/serde structures.

---

# 49. MODEL SOURCE SECURITY

Support model revisions.

Where remote model download exists:

```text
pin model revision
verify downloaded files
prevent directory traversal
enforce download limits
```

Model identity should include revision/hash.

---

# 50. EXTERNAL API

Expose:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
```

Support SSE streaming.

Example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "...",
    "messages": [
      {"role":"user","content":"hello"}
    ],
    "stream": true
  }'
```

Keep API implementation independent of scheduler internals.

---

# 51. SAMPLING

Sampling configuration includes:

```text
temperature
top_p
top_k
max_tokens
seed
stop tokens
```

For v1 perform final sampling on the last stage or coordinator.

Choose one owner deterministically.

Never sample independently on multiple machines.

---

# 52. TOKENIZER

Tokenizer identity belongs to the model revision.

Do not permit different nodes to silently use different tokenizer versions.

Coordinator should own request tokenization initially unless there is a strong reason otherwise.

---

# 53. REPRODUCIBILITY

When:

```text
temperature = 0
```

distributed output should match single-node output within the numerical constraints of the runtime.

Create regression tests comparing:

```text
single-node
vs
distributed
```

for the same model/input.

---

# 54. OBSERVABILITY

Use structured tracing.

Track each request across every node.

Every request gets:

```text
trace_id
request_id
sequence_id
```

Metrics must include at least:

```text
tokens/sec
time-to-first-token
inter-token latency
prefill duration
decode duration

stage compute time
stage idle time

bytes transferred
network RTT
network throughput

memory usage
KV usage
model weight memory

cache hit/miss

request queue depth

node health

planner predicted latency
actual latency
prediction error
```

The prediction error metric is especially important.

The scheduler should become measurable.

---

# 55. DEBUGGING

Provide:

```bash
aether doctor
```

It must validate:

```text
node connectivity
authentication
runtime availability
model files
memory
transport
port reachability
clock sanity
MLX functionality
peer compatibility
```

Output actionable errors.

Bad:

```text
RPC failed
```

Good:

```text
m5-air cannot establish QUIC data-plane connection to m4-air:9001.

Control-plane connection succeeds.
UDP/9001 appears blocked.

Try:
...
```

---

# 56. SIMULATOR

This is mandatory.

Create an in-process simulator where fake nodes can be defined:

```yaml
nodes:
  - name: m4
    memory_gb: 16
    compute_score: 1.0

  - name: m5
    memory_gb: 16
    compute_score: 1.3

links:
  - from: m4
    to: m5
    bandwidth_gbps: 8
    latency_ms: 0.3
```

Then run:

```bash
aether-sim plan model.json topology.yaml
```

The simulator lets us test the planner without owning 50 physical machines.

Support virtual:

```text
slow nodes
fast nodes
asymmetric links
network partitions
memory shortages
disk shortages
node failures
thermal degradation
```

---

# 57. FAKE RUNTIME

Before MLX execution is complete, implement:

```text
FakeRuntimeBackend
```

It should simulate transformer stages deterministically.

Use it to validate:

```text
protocol
pipeline
scheduler
networking
failure handling
cancellation
```

Do not couple distributed-systems development to GPU kernel progress.

---

# 58. TESTING

Use:

```text
unit tests
integration tests
property-based tests
chaos tests
benchmarks
```

Planner property tests must verify invariants such as:

```text
no node exceeds memory
every model component has exactly one owner where appropriate
no layer missing
no layer duplicated accidentally
pipeline ordering valid
all boundaries have transport paths
unsupported runtime rejected
```

---

# 59. PLANNER EDGE CASES

Test at least:

```text
model fits completely on one node

model barely fits one node

model only fits distributed

model cannot fit cluster

zero free-memory node

node faster but memory-constrained

node slower but huge memory

very slow network

high bandwidth/high RTT network

low bandwidth/low RTT network

asymmetric link

one huge transformer layer

huge embedding table

huge LM head

tied embedding/head

different layer sizes

different layer compute costs

thermal throttling

node memory changes after plan

node joins during generation

node leaves during generation

model revision changes

runtime versions differ
```

---

# 60. TRANSPORT EDGE CASES

Test:

```text
connection drops mid-tensor

partial frame

corrupt checksum

duplicate frame

stale frame

oversized tensor claim

zero-length payload

wrong dtype

wrong shape

wrong plan epoch

unexpected stage sender

slow receiver

concurrent requests

stream cancellation

peer reconnect
```

Never panic due to network input.

---

# 61. MODEL EDGE CASES

Test:

```text
missing config

missing tensor

duplicate tensor

unknown dtype

unsupported quantization

invalid safetensors offsets

truncated checkpoint

bad index JSON

incorrect tensor dimensions

unknown architecture

zero-layer model

shared/tied tensors

checkpoint split across many files
```

Return structured errors.

---

# 62. KV EDGE CASES

Test:

```text
cache full

cache block pinned during eviction

same prefix from two requests

corrupt remote block

remote storage unavailable

promotion race

duplicate block writes

request cancelled during KV write

lost KV owner

session resumed after eviction

context length exceeds planned capacity
```

---

# 63. ARITHMETIC SAFETY

Memory and tensor dimensions come from external files.

Never write:

```rust
let bytes = a * b * c;
```

without considering overflow.

Use checked calculations for:

```text
tensor element counts
tensor byte sizes
KV estimates
network frame lengths
offset ranges
```

Malformed models must produce errors, not integer wraparound.

---

# 64. RUST QUALITY

Prefer:

```text
typed IDs
enums
Result
thiserror
RAII
bounded channels
CancellationToken
Arc where genuinely shared
```

Avoid:

```text
unwrap()
expect()
global mutable state
giant Arc<Mutex<Everything>>
unbounded mpsc
stringly typed state machines
```

`unwrap()` is acceptable in tests where failure is intentional and obvious.

Library crates should expose typed errors.

Use `anyhow` mainly at binary/application boundaries.

---

# 65. CONCURRENCY

Use Tokio.

Every long-running subsystem should have explicit lifecycle management.

Use structured cancellation.

Shutdown order should gracefully:

```text
stop accepting API requests
drain active requests
cancel remaining work after timeout
flush relevant state
close peer connections
unload runtime
exit
```

---

# 66. PERSISTENCE

Persist at minimum:

```text
node identity
cluster identity
known peers
model metadata
benchmark results
planner observations
model cache index
```

Do not persist active mutex state or runtime handles.

Use schema migrations.

Treat persistent files as potentially corrupted.

---

# 67. VERSION COMPATIBILITY

Handshake includes:

```text
protocol version
daemon version
runtime type
runtime version
supported capabilities
```

Nodes with incompatible protocol versions must fail clearly.

Do not let incompatible peers discover the problem halfway through inference.

---

# 68. FEATURE NEGOTIATION

Never infer feature support from version number alone.

Exchange capability flags.

Example:

```text
PIPELINE_V1
KV_BLOCKS_V1
QUIC_DATA_V1
TP_V1
RDMA_V1
```

---

# 69. PLAN EXPLANATION

Every scheduler decision must be explainable.

Provide:

```bash
aether plan model --explain
```

Example:

```text
Rejected split after layer 29:
  M5 estimated peak = 14.3 GB
  usable = 13.2 GB

Rejected tensor parallel:
  measured RTT = 2.7 ms
  exceeds TP policy threshold

Selected layer 24 boundary:
  M4 stage = 81 ms
  M5 stage = 79 ms
  memory headroom = 18%
```

This is an important product feature.

---

# 70. PLANNER DRY RUN

Allow:

```bash
aether plan ...
```

without downloading/loading the entire model where metadata is sufficient.

Separate:

```text
planning
```

from:

```text
execution
```

---

# 71. REPLANNING

Do not automatically migrate active inference whenever a better node appears.

Replanning should first affect:

```text
new requests
```

Model migration is expensive.

Use hysteresis so plans do not oscillate when node performance fluctuates slightly.

---

# 72. RESOURCE RESERVATION

A scheduler plan is not sufficient.

Reserve memory before loading.

Flow:

```text
planner
 ↓
tentative plan
 ↓
reserve resources on workers
 ↓
all workers ACK
 ↓
commit plan
```

If one reservation fails:

```text
release all reservations
replan
```

Use timeout-based leases so abandoned reservations disappear.

---

# 73. LOAD RACES

Two simultaneous API requests must not cause two independent copies of the same model partition to load accidentally.

Use per-model/per-partition loading state.

Concurrent callers await the same load operation.

---

# 74. WEIGHT TRANSFER

Never transfer weights for every request.

Weight placement is cluster state.

Weight transfers happen on:

```text
initial load
replan
node replacement
cache miss
model update
```

Activation transfer happens during inference.

Keep these concepts separate.

---

# 75. LOCAL WEIGHT CACHE

Before transferring weights over network:

```text
check local model cache
verify hash
reuse if valid
```

Only fetch missing ranges/files.

---

# 76. WEIGHT PARTITIONING

Where checkpoint files contain tensors from multiple assigned stages, do NOT necessarily rewrite checkpoints immediately.

Initially allow workers to:

```text
memory-map checkpoint
materialize only required tensors
```

Later optimize by generating partition packs.

Measure before implementing custom packed formats.

---

# 77. PERFORMANCE BASELINE

Before claiming distributed speedups, benchmark:

```text
single-node model that fits

distributed pipeline

prefill latency

decode tok/s

TTFT

memory usage

network traffic
```

The goal of the initial distributed experiment may be:

> run a model that otherwise cannot fit

rather than:

> make generation twice as fast.

Report this honestly.

---

# 78. MILESTONE 0 — SKELETON

Build:

```text
Rust workspace
aether CLI
aetherd daemon
node identities
config
structured tracing
clean shutdown
```

Acceptance:

```text
two daemons start
CLI sees both
```

---

# 79. MILESTONE 1 — CLUSTER DISCOVERY

Implement:

```text
local discovery
manual join
heartbeats
capability exchange
```

Acceptance:

```bash
aether nodes
```

shows both Macs.

---

# 80. MILESTONE 2 — TOPOLOGY PROFILER

Measure:

```text
RTT
bandwidth
memory
disk
runtime capability
```

Acceptance:

```bash
aether topology
```

returns measured edges.

---

# 81. MILESTONE 3 — MODEL INSPECTOR

Support MLX/SafeTensors dense transformer checkpoints.

Output:

```text
model architecture
layers
tensor map
weight bytes/layer
KV metadata
quantization
```

Acceptance:

```bash
aether model inspect <model>
```

works without loading full model into accelerator memory.

---

# 82. MILESTONE 4 — PLANNER

Implement dense contiguous pipeline planning.

Start with:

```text
two nodes
```

then generalize to N.

Acceptance:

```bash
aether plan <19GB-model>
```

returns a valid automatic split for two 16 GB Macs or explains why no split works.

---

# 83. MILESTONE 5 — FAKE DISTRIBUTED INFERENCE

Use FakeRuntime.

Implement:

```text
stage execution
activation transport
token loop
request IDs
cancellation
failure handling
```

Acceptance:

```text
distributed deterministic fake model produces correct output
```

---

# 84. MILESTONE 6 — MLX SINGLE NODE

Implement Rust MLX backend.

Before distributed execution, prove:

```text
load supported model
tokenize prompt
prefill
decode
sample
stream text
```

on one machine.

Create regression tests.

---

# 85. MILESTONE 7 — MLX PARTITION LOADING

Each worker loads only its assigned transformer stages.

Acceptance:

```text
M4 process does not load M5-owned layer tensors
M5 process does not load M4-owned layer tensors
```

Measure actual resident memory to verify this.

---

# 86. MILESTONE 8 — TWO-MAC PIPELINE

Execute:

```text
M4 stage
 ↓ activation
M5 stage
 ↓ logits
```

for prefill and decode.

Acceptance:

A model too large for one 16 GB Mac runs across two 16 GB Macs.

---

# 87. MILESTONE 9 — KV MANAGER

Add:

```text
per-stage KV ownership
block manager
memory budget
local cold tier
prefix infrastructure
```

Do not start remote SSD KV until local semantics are correct.

---

# 88. MILESTONE 10 — FAILURE RECOVERY

Inject failures:

```text
kill worker
disconnect link
fill disk
cancel request
```

Verify cleanup and correct errors.

---

# 89. MILESTONE 11 — MULTI-REQUEST SCHEDULING

Add:

```text
bounded request queue
microbatching where runtime permits
pipeline concurrency
fairness
```

Measure throughput improvements separately from latency.

---

# 90. FUTURE MILESTONES

Only after pipeline execution is stable:

```text
tensor parallel MLX groups

Thunderbolt/RDMA transport

CUDA backend

NCCL groups

ROCm

mixed runtime pipeline

remote KV tiers

prefix-cache federation

replicated model stages

automatic failover

multi-coordinator consensus

multi-tenant scheduling

energy-aware scheduling
```

---

# 91. CRATE GUIDANCE

Use well-established Rust ecosystem components where appropriate.

Candidates include:

```text
tokio
tonic
prost
axum
tower
quinn
serde
serde_json
tracing
OpenTelemetry
sysinfo
safetensors
thiserror
anyhow
bytes
dashmap where justified
blake3
```

Do not blindly install these.

Before adding any dependency:

1. verify its current maintained version;
2. inspect its license;
3. confirm platform support;
4. confirm MSRV compatibility;
5. explain why it is needed.

Keep dependencies minimal.

---

# 92. NO PREMATURE OPTIMIZATION

Do not write custom:

```text
Metal kernels
CUDA kernels
RDMA implementation
tensor serialization format
consensus algorithm
```

before existing components have been measured and shown insufficient.

First build a correct measurable system.

Then optimize bottlenecks.

---

# 93. DOCUMENTATION

Maintain:

```text
docs/architecture.md
docs/protocol.md
docs/planner.md
docs/model-runtime.md
docs/kv-cache.md
docs/failure-model.md
docs/security.md
```

Include diagrams.

Document invariants, not just APIs.

---

# 94. ARCHITECTURAL INVARIANTS

Keep these true:

### Invariant 1

Exactly one committed execution plan owns a request.

### Invariant 2

Every required model component has a valid owner.

### Invariant 3

A worker never executes work from an expired plan epoch.

### Invariant 4

No scheduler-approved placement exceeds its declared memory budget.

### Invariant 5

Hot KV is colocated with its compute stage unless explicitly executing a supported migration/offload operation.

### Invariant 6

A generated token is only committed after the complete forward pass and sampling operation succeeds.

### Invariant 7

Untrusted network metadata is validated before memory allocation.

### Invariant 8

Weight and KV corruption never silently influences generation.

---

# 95. FIRST REAL-WORLD TARGET

Optimize the first end-to-end implementation for this scenario:

```text
Node A
Apple Silicon M4
16 GB unified RAM

Node B
Apple Silicon M5
16 GB unified RAM

Dense MLX model
~19 GB weights

Model cannot safely fit on a single machine.

AetherMesh automatically:

1. discovers both Macs;
2. authenticates them;
3. benchmarks hardware;
4. benchmarks network;
5. inspects checkpoint;
6. determines memory requirements;
7. profiles/estimates layer compute;
8. evaluates legal layer boundaries;
9. selects an optimal split;
10. reserves resources;
11. ensures weight availability;
12. loads each partition;
13. establishes pipeline;
14. performs prefill;
15. maintains local KV for owned layers;
16. transfers boundary activations;
17. generates tokens;
18. streams output through OpenAI-compatible API;
19. records prediction-vs-actual metrics;
20. handles cancellation and failures cleanly.
```

---

# 96. WHAT NOT TO DO

Do NOT:

```text
hard-code 50/50 split

hard-code M4/M5 assumptions

assume layer sizes equal

assume layer compute equal

assume network symmetric

assume total RAM is usable RAM

put hot KV on arbitrary remote SSD

transfer weights per generated token

load the whole model on every machine

use Python subprocesses as the architecture

silently fall back to CPU

panic on malformed peer traffic

trust LAN peers automatically

use unlimited queues

hide scheduler reasoning

claim distributed execution is faster without benchmarking

start with tensor parallelism

start with 10 hardware platforms
```

---

# 97. DEVELOPMENT METHOD

Implement one vertical slice at a time.

For every milestone:

1. write the relevant types;
2. write invariants;
3. write unit tests;
4. implement functionality;
5. write integration tests;
6. add metrics;
7. run benchmarks;
8. document limitations;
9. commit only when tests pass.

Do not produce huge speculative modules full of TODO implementations.

Prefer small working increments.

---

# 98. WHEN SOMETHING IS UNKNOWN

Do not invent behavior for:

```text
MLX
mlx-c
Apple Metal
SafeTensors
QUIC
Rust crates
model architecture
quantization
hardware APIs
```

Read the current upstream source/documentation first.

If upstream functionality is unavailable:

1. state what is missing;
2. isolate the missing capability behind an interface;
3. implement the smallest correct fallback;
4. continue without corrupting architecture.

---

# 99. INITIAL TASK

Start by producing:

```text
1. final Rust workspace architecture
2. major Rust types
3. protocol definitions
4. node state machine
5. request state machine
6. model state machine
7. planner data model
8. memory estimator
9. two-node dense partition algorithm
10. fake runtime design
11. implementation order
```

Then begin implementing **Milestone 0**.

Do not implement MLX inference before the cluster, model manifest, planner, fake runtime, and transport abstractions are structurally sound.

The first important correctness milestone is:

> Given two virtual heterogeneous machines and a dense model manifest, AetherMesh can prove whether the model fits and produce the best valid contiguous layer partition.

The first important physical milestone is:

> Two real Apple Silicon Macs can automatically discover each other and run one dense MLX model that cannot fit safely on either machine alone.

Build toward those two milestones relentlessly.

# 100. DEFINITION OF DONE FOR V1

AetherMesh v1 is complete when this works:

```bash
# Mac 1
aether init
aetherd

# Mac 2
aether join <mac-1>
aetherd
```

Then:

```bash
aether nodes
```

shows both nodes.

Then:

```bash
aether plan <model>
```

automatically produces a valid placement.

Then:

```bash
aether serve <model>
```

loads the distributed model.

And:

```bash
curl localhost:8080/v1/chat/completions ...
```

streams correct generated tokens from a model whose required runtime memory would prevent safe single-machine execution.

No manual:

```text
layer mapping
tensor split
node ranking
weight placement
KV ownership
```

should be required from the user.

The core product promise is:

> **Give AetherMesh machines and a model. AetherMesh figures out how to run it.**
:::

A few choices in that prompt are deliberate: current llama.cpp RPC explicitly warns its remote RPC backend is fragile/insecure, so AetherMesh treats peer authentication and encrypted transport as first-class rather than an afterthought. citeturn658496search5 Current vLLM tiering also reinforces the hot→RAM→secondary-storage hierarchy instead of treating SSD as active KV memory. citeturn658496search0turn658496search2

The **simulator + fake runtime** parts are especially important. They let you develop the genuinely novel piece—the automatic planner—on one laptop before needing a pile of physical hardware.

---

## User

Now explain me in simple eng how will it handle Nvidia + apple + x86 when all this stack is in the same pool and model is around 32 GB of model

---

## Assistant

Think of the pool as **one cluster, but not one uniform GPU**.

Say you have:

```text
NVIDIA PC
  RTX GPU
  16 GB VRAM
  CUDA

Mac
  Apple Silicon
  16 GB unified memory
  MLX / Metal

x86 PC
  64 GB RAM
  CPU only
```

and a dense model with roughly **32 GB of weights**.

AetherMesh first asks:

```text
What can each machine actually do?
```

It discovers:

```text
NVIDIA → CUDA compute
Apple  → MLX compute
x86    → CPU compute + lots of RAM + SSD
```

Then it creates **runtime islands**:

```text
CUDA island
   NVIDIA

MLX island
   Apple

CPU island
   x86
```

The key point is:

> We do NOT pretend NVIDIA + Apple + x86 are three identical GPUs.

They are three different execution environments.

For a dense model, the easiest way to combine them is **pipeline parallelism**.

For example, suppose the model has 64 layers.

A possible plan could be:

```text
            32 GB model
                 │
        ┌────────┴────────┐

NVIDIA
CUDA
Layers 0–27
~13 GB weights
       │
       │ activation
       ▼
Apple
MLX
Layers 28–51
~11 GB weights
       │
       │ activation
       ▼
x86 CPU
Layers 52–63
~8 GB weights
       │
       ▼
     output
```

So every generated token travels:

```text
NVIDIA
   ↓
Apple
   ↓
x86
   ↓
next token
```

Each machine only stores the weights for the layers it owns.

---

But this probably **wouldn't be the optimal split**.

The x86 CPU might be dramatically slower than the NVIDIA GPU and Apple GPU.

Suppose benchmarks say:

```text
NVIDIA layer:  1 ms
Apple layer:   2 ms
x86 CPU layer: 25 ms
```

If you give x86 12 layers, the entire cluster becomes bottlenecked by the x86 machine.

So your scheduler may instead decide:

```text
NVIDIA
Layers 0–34
~15 GB

Apple
Layers 35–63
~13 GB

x86
NO MODEL COMPUTE
```

And x86 becomes:

```text
model storage
cold KV storage
prefix cache
checkpoint cache
request coordinator
possibly tokenizer/API server
```

That is completely valid.

**Being in the pool does not mean every machine must calculate every token.**

That is one of the most important concepts for your project.

---

### How do NVIDIA and Apple exchange data?

You cannot send an MLX tensor directly into CUDA.

So AetherMesh needs a runtime-neutral tensor format at the boundary.

Internally:

```text
NVIDIA CUDA tensor
       │
       │ extract activation
       ▼
AetherMesh Tensor
{
    shape: [1, 1, 5376]
    dtype: bf16
    bytes: ...
}
       │
       │ network
       ▼
Apple
       │
       │ create MLX array
       ▼
MLX tensor
```

And in the opposite direction:

```text
MLX
 ↓
neutral tensor bytes
 ↓
network
 ↓
CUDA
```

You're not converting the **weights** every token.

Only the relatively small intermediate activation gets transferred.

---

### What about the weights themselves?

This is another important layer of the system.

Imagine you download:

```text
Gemma
original checkpoint
32 GB
```

AetherMesh has a logical model representation:

```text
embedding
layer.0
layer.1
...
layer.63
norm
lm_head
```

Then each runtime gets its own representation.

For example:

```text
Original checkpoint
        │
        ├──────────────► CUDA representation
        │                layers 0–34
        │
        └──────────────► MLX representation
                         layers 35–63
```

Potentially:

```text
x86 CPU
  ↓
stores master checkpoint

NVIDIA asks:
"Give me tensors for layers 0–34"

Apple asks:
"Give me tensors for layers 35–63"
```

Each backend converts/caches them into the format it prefers.

Eventually you might have:

```text
/model-cache/

gemma/
  source/
      safetensors/

  cuda/
      quantized-format-A/

  mlx/
      mlx-format-B/

  cpu/
      gguf/
```

Same logical model.

Different optimized representations.

---

### Now imagine the NVIDIA has only 8 GB VRAM

Then perhaps:

```text
RTX
8GB VRAM

Mac
16GB unified

x86
64GB RAM
```

The planner could calculate:

```text
usable RTX memory:   ~6 GB
usable Mac memory:   ~12 GB

total fast memory:
~18 GB

model:
32 GB
```

It cannot fit.

Now the scheduler has several choices.

First it could look for a more aggressive quantization:

```text
32 GB FP16/BF16
       ↓
Q8
~17 GB
```

Then suddenly:

```text
RTX + Mac
```

may be enough.

This is something I think AetherMesh should eventually do automatically:

```text
Model requested

FP16?
❌ does not fit

Q8?
✅ fits

Q6?
✅ fits

Q4?
✅ fits

quality/performance policy:
choose Q8
```

That's a very powerful part of the product.

---

If quantization isn't allowed, then it might have to use x86 RAM:

```text
RTX
layers 0–10
6 GB

Mac
layers 11–32
12 GB

x86
layers 33–63
14 GB
```

This works.

But:

```text
Performance warning:

CPU stage estimated to account for
72% of decode latency.
```

AetherMesh should explicitly tell you:

> It is **possible**, but not necessarily **good**.

---

### KV works similarly

Suppose:

```text
NVIDIA owns layers 0–34

Apple owns layers 35–63
```

Then:

```text
NVIDIA
KV for layers 0–34

Apple
KV for layers 35–63
```

The x86 machine might have:

```text
64GB RAM
2TB SSD
```

So it can hold:

```text
cold prefix KV
paused conversations
old session KV
model checkpoints
```

But during active generation:

```text
NVIDIA KV stays NVIDIA-side

Apple KV stays Apple-side
```

We don't want:

```text
GPU → network → x86 SSD → network → GPU
```

on every token.

That would be painfully slow.

---

## Your scheduler basically asks these questions

When the 32 GB model arrives:

```text
1. Can one node run it?

NVIDIA?
No.

Apple?
No.

x86?
Technically yes maybe,
but CPU performance terrible.
```

Then:

```text
2. Can compatible nodes run it together?

2 NVIDIA GPUs?
maybe

multiple Macs?
maybe
```

Then:

```text
3. Can heterogeneous pipeline execution run it?

NVIDIA + Apple?
yes

NVIDIA + Apple + CPU?
yes, but slower
```

Then it evaluates candidates:

```text
PLAN A
NVIDIA + Apple

Doesn't fit.
❌


PLAN B
NVIDIA + Apple
Q8 model

Predicted: 18 tok/s
✅


PLAN C
NVIDIA + Apple + x86
full precision

Predicted: 2.5 tok/s
✅


PLAN D
x86 only

Predicted: 0.7 tok/s
✅
```

And chooses:

```text
PLAN B
```

because it's the best policy-compatible option.

---

So conceptually your whole cluster might look like this:

```text
                  AetherMesh
                      │
                 Global Pool
                      │
        ┌─────────────┼─────────────┐
        │             │             │
     NVIDIA          Apple         x86
     CUDA            MLX           CPU
        │             │             │
        └─────┐   ┌───┘             │
              │   │                 │
          FAST COMPUTE          STORAGE /
              │                 COLD KV /
              │                 FALLBACK
              └──────┬──────────────┘
                     │
                32 GB MODEL
```

And the critical design philosophy is:

> **The pool is heterogeneous, but each device gets the job it is best suited for.**

Not:

> "I have three machines, therefore each gets 10.6 GB."

Sometimes the optimal result might even be:

```text
NVIDIA: 70% of weights
Apple:  30%
x86:    0%
```

while x86 still contributes enormously by providing storage/cache/control-plane resources.

That dynamic decision-making is probably the most valuable part of AetherMesh.

---

## User

Give me actual APIs of NVIDIA MLX and all the things that you are going to use so that I can know how you will transfer activations from one machine to another machine and convert as well.

---

## Assistant

I’m going to pin this down to the concrete call path: **CUDA tensor → host staging buffer → network frame → MLX array**, plus the reverse direction, and I’ll separate the APIs we can use today from things that only work within NVIDIA-only or Apple-only islands.

---

## Assistant

The key thing is that **there is no API that directly converts a CUDA tensor into an MLX tensor across two machines**. CUDA/NCCL and MLX Distributed are separate worlds. For the NVIDIA↔Apple boundary, AetherMesh itself defines a small tensor wire protocol and moves the raw activation bytes; each runtime reconstructs its native tensor on the receiving side.

---

## Assistant

Yes. The clean way is to define **one AetherMesh tensor format on the wire**, then have a CUDA adapter, MLX adapter, and CPU adapter around it.

The actual stack I would start with is:

| Part | Implementation/API |
|---|---|
| NVIDIA execution | CUDA through Rust `cudarc` |
| NVIDIA GEMMs | cuBLAS/cuBLASLt via `cudarc` |
| NVIDIA custom kernels | CUDA PTX/NVRTC through `cudarc` |
| Apple execution | `mlx-c` through Rust FFI |
| Apple↔Apple | MLX Distributed when useful |
| NVIDIA↔NVIDIA | NCCL when useful |
| NVIDIA↔Apple/x86 | **our own tensor transport** |
| Network transport | Rust `quinn` / QUIC |
| CPU/x86 | Rust CPU backend, likely Candle initially |
| Model storage | SafeTensors |
| Control plane | Tokio + tonic/protobuf |

The important distinction is that **NCCL and MLX Distributed are optimizations inside compatible islands**. They do not solve NVIDIA↔Apple.

---

## 1. NVIDIA → Apple: actual path

Assume the NVIDIA stage finishes layer 30 and produces:

```text
CUDA GPU activation

shape = [1, 1, 5376]
dtype = BF16
size ≈ 10.5 KB
```

It physically lives in NVIDIA VRAM.

### Step A — CUDA tensor

In Rust using `cudarc`, we'd have something conceptually like:

```rust
let ctx = CudaContext::new(0)?;
let stream = ctx.default_stream();

let activation: CudaSlice<bf16> = ...;
```

`cudarc` currently exposes `CudaContext`, `CudaStream`, `CudaSlice<T>`, `alloc_pinned`, `memcpy_dtoh`, `memcpy_htod`, kernel loading, etc. citeturn584717search0turn644441view1

Underneath, those correspond to CUDA operations such as:

```c
cudaStreamCreate(...)

cudaHostAlloc(...)

cudaMemcpyAsync(
    host_ptr,
    gpu_ptr,
    bytes,
    cudaMemcpyDeviceToHost,
    stream
)
```

CUDA explicitly provides `cudaMemcpyAsync` for asynchronous host/device copies, and pinned memory allocated through `cudaHostAlloc` is intended for efficient host↔device staging. citeturn300986view1turn217080search0

In Rust we'd probably maintain a reusable pinned buffer:

```rust
let mut staging =
    unsafe { ctx.alloc_pinned::<u16>(activation_len)? };

stream.memcpy_dtoh(&activation, &mut staging)?;
```

`cudarc::PinnedHostSlice` is page-locked CUDA-visible host memory and is designed specifically for this `memcpy_dtoh`/`memcpy_htod` path. citeturn842429view0

So physically:

```text
RTX VRAM
    │
    │ cudaMemcpyAsync D2H
    ▼
Pinned system RAM
```

---

# 2. We don't convert BF16 into some special network datatype

We send it essentially as:

```text
HEADER

magic
protocol_version

request_id
sequence_id
token_position
stage_id

dtype = BF16

rank = 3
shape = [1, 1, 5376]

payload_len = 10752

checksum
```

followed by:

```text
10752 raw bytes
```

So internally something like:

```rust
struct TensorHeader {
    request_id: u128,
    sequence_id: u64,

    stage: u32,
    token_position: u64,

    dtype: WireDType,
    shape: Vec<u64>,

    byte_len: u64,
}
```

And:

```rust
enum WireDType {
    BF16,
    F16,
    F32,
}
```

I'd probably force **row-major contiguous BF16** as the first cross-runtime representation.

That makes the boundary predictable.

---

# 3. Send those bytes using QUIC

With Quinn:

```rust
let mut send = connection.open_uni().await?;

send.write_all(&header).await?;
send.write_all(payload).await?;
send.finish()?;
```

On Apple:

```rust
let mut recv = connection.accept_uni().await?;

recv.read_exact(&mut header).await?;
recv.read_exact(&mut payload).await?;
```

Those are real Quinn APIs today: `Connection::open_uni()`, `accept_uni()`, `SendStream::write_all()` and `RecvStream::read_exact()`. citeturn556476search0turn556476search2turn556476search3

So:

```text
NVIDIA machine

VRAM
 ↓
Pinned RAM
 ↓
QUIC

================ NETWORK ================

QUIC
 ↓
Mac RAM
```

---

# 4. Mac: raw network bytes → MLX array

This is where `mlx-c` comes in.

The straightforward safe v1 API is:

```c
mlx_array mlx_array_new_data(
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype
);
```

That API explicitly accepts an existing buffer and copies it into an MLX array. citeturn335508view0

So from our Rust FFI:

```rust
let shape = [1i32, 1, 5376];

let x = mlx_array_new_data(
    payload.as_ptr().cast(),
    shape.as_ptr(),
    3,
    MLX_BFLOAT16,
);
```

Now:

```text
received network bytes
       ↓
mlx_array_new_data()
       ↓
MLX array
       ↓
Metal / Apple GPU
```

And the Mac continues:

```text
Layer 31
Layer 32
Layer 33
...
```

There is **no CUDA→MLX mathematical conversion** here.

We're doing:

```text
CUDA BF16 tensor
      ↓
BF16 bytes
      ↓
MLX BF16 tensor
```

Same numbers.

Different tensor owner.

---

# 5. Why `mlx_array_new_data()` is really useful

The current MLX-C API exposes the dtype enum directly:

```c
MLX_FLOAT16
MLX_FLOAT32
MLX_BFLOAT16
...
```

and exposes array creation, shapes, byte sizes, dtype queries and raw data access. citeturn335508view0

For example:

```c
mlx_array_dtype(arr);
mlx_array_shape(arr);
mlx_array_nbytes(arr);
mlx_array_ndim(arr);
```

So the adapter doesn't need Python anywhere.

---

# 6. Apple → NVIDIA is even more interesting

Suppose the Apple machine finishes its stage:

```text
MLX Array
shape = [1, 1, 5376]
dtype = BF16
```

MLX computation is lazy.

So first:

```c
mlx_array_eval(array);
```

MLX-C explicitly exposes this API. citeturn335508view0

Before exposing raw bytes, I would make the output contiguous:

```c
mlx_contiguous(
    &contiguous,
    array,
    false,
    stream
);
```

`mlx_contiguous` exists in MLX-C's operations API. citeturn335508view1

Then:

```c
mlx_array_eval(contiguous);
```

Now retrieve:

```c
size_t bytes = mlx_array_nbytes(contiguous);
mlx_dtype dtype = mlx_array_dtype(contiguous);

const bfloat16_t* data =
    mlx_array_data_bfloat16(contiguous);
```

MLX-C documents that the raw `mlx_array_data_*` functions require the array to have been evaluated first. citeturn335508view0

Because Apple Silicon uses unified memory, MLX's CPU and GPU operate within its shared-memory model rather than a CUDA-style discrete VRAM/host-RAM split. citeturn231893search2

So conceptually:

```text
Apple GPU
   │
   │ unified memory
   ▼
MLX array bytes
   │
   ▼
QUIC
```

We don't need an NVIDIA-style GPU→host PCIe copy first.

---

# 7. NVIDIA receives it

RTX machine receives:

```text
BF16 network bytes
```

into our pinned staging buffer.

Then:

```rust
stream.memcpy_htod(
    &staging,
    &mut cuda_activation
)?;
```

Underneath that's effectively:

```c
cudaMemcpyAsync(
    gpu_ptr,
    host_ptr,
    bytes,
    cudaMemcpyHostToDevice,
    stream
);
```

The CUDA docs support asynchronous H2D/D2H copies, while `cudarc` exposes them as `memcpy_htod()` / `memcpy_dtoh()`. citeturn300986view1turn584717search1

Therefore:

```text
Mac MLX
   │
   ▼
raw BF16 bytes
   │
 QUIC
   │
   ▼
NVIDIA pinned RAM
   │
 cudaMemcpyAsync H2D
   ▼
NVIDIA VRAM
   │
   ▼
next CUDA layer
```

That's the entire heterogeneous pipeline.

---

# The full real path

```text
                 NVIDIA MACHINE

            CUDA transformer layer
                     │
                     ▼
             CudaSlice<BF16>
                     │
          memcpy_dtoh / cudaMemcpyAsync
                     │
                     ▼
               pinned RAM
                     │
                     ▼
            Aether Tensor Frame
        BF16 + shape + raw payload
                     │
                     ▼
              Quinn / QUIC
                     │
                     │
        ═════════ NETWORK ═════════
                     │
                     ▼
               Quinn / QUIC
                     │
                     ▼
               receive bytes
                     │
                     ▼
           mlx_array_new_data()
                     │
                     ▼
               MLX BF16 array
                     │
                     ▼
          Apple transformer layer
```

And backwards:

```text
MLX array
   ↓
mlx_contiguous
   ↓
mlx_array_eval
   ↓
mlx_array_data_bfloat16
   ↓
QUIC
   ↓
CUDA pinned buffer
   ↓
memcpy_htod
   ↓
CudaSlice<BF16>
```

---

# What about dtype differences?

Suppose CUDA outputs:

```text
BF16
```

but some MLX stage is running:

```text
FP16
```

We have two choices.

Bad choice:

```text
CUDA BF16
 ↓
CPU converts BF16→FP16
 ↓
network
```

I'd avoid that.

Better:

```text
CUDA BF16
 ↓
network BF16
 ↓
MLX BF16
 ↓
mlx_astype(..., MLX_FLOAT16, ...)
```

MLX-C exposes:

```c
mlx_astype(
    mlx_array* res,
    mlx_array a,
    mlx_dtype dtype,
    mlx_stream stream
);
```

citeturn335508view1

But ideally the scheduler makes both sides use the **same boundary dtype**, so there's no cast at all:

```text
AetherMesh boundary dtype = BF16
```

---

# NVIDIA ↔ NVIDIA is different

If there are:

```text
RTX 4090 A
RTX 4090 B
```

we should generally NOT do:

```text
GPU → CPU → QUIC → CPU → GPU
```

if we can avoid it.

Use NCCL.

Real APIs:

```c
ncclSend(
    sendbuff,
    count,
    datatype,
    peer,
    comm,
    stream
);

ncclRecv(
    recvbuff,
    count,
    datatype,
    peer,
    comm,
    stream
);
```

NCCL can operate directly on CUDA device buffers. citeturn231893search0

So NVIDIA island:

```text
CUDA GPU A
     │
     │ NCCL
     ▼
CUDA GPU B
```

No AetherMesh tensor conversion required internally.

---

# Apple ↔ Apple is also different

MLX-C already exposes:

```c
mlx_distributed_send(...)
mlx_distributed_recv(...)
mlx_distributed_recv_like(...)
```

as well as:

```c
mlx_distributed_all_sum(...)
mlx_distributed_all_gather(...)
mlx_distributed_sum_scatter(...)
```

citeturn406746view0

And distributed setup APIs include:

```c
mlx_distributed_init(...)
mlx_distributed_group_rank(...)
mlx_distributed_group_size(...)
mlx_distributed_group_split(...)
```

citeturn406746view1

So two Macs may use:

```text
MLX array
   │
MLX distributed send
   │
   ▼
MLX array
```

instead of our generic host serialization.

---

# Therefore the transport selector becomes

```rust
match (source.runtime, destination.runtime) {

    (CUDA, CUDA)
        if nccl_compatible =>
            NCCL,

    (MLX, MLX)
        if mlx_distributed_compatible =>
            MLXDistributed,

    _ =>
        AetherWireQUIC,
}
```

This is exactly how I'd design it.

---

# x86 is the easiest boundary

CPU-only x86:

```text
network bytes
     ↓
RAM
     ↓
CPU tensor
```

There isn't necessarily a GPU-copy step at all.

If we use Candle for the CPU backend, current Candle supports constructing CPU tensors with APIs such as:

```rust
Tensor::from_vec(data, shape, &Device::Cpu)
```

and explicitly notes that `from_vec` does not need an additional data copy when the target device is CPU. citeturn662722search0

Candle also has SafeTensors/raw-buffer conversion handling for BF16, F16, F32 and other dtypes. citeturn662722search1

So:

```text
NVIDIA
   ↓
AetherWire
   ↓
x86 CPU

or

Apple
   ↓
AetherWire
   ↓
x86 CPU
```

works with the same wire format.

---

## One important thing we should NOT use: DLPack

At first DLPack sounds perfect:

```text
CUDA tensor → DLPack → MLX
```

but across machines it doesn't solve the problem.

DLPack essentially describes a tensor and its memory. A CUDA memory pointer from:

```text
Machine A
```

means absolutely nothing on:

```text
Machine B
```

So DLPack is useful for **same-process/same-machine interoperability**, not general distributed transport.

Same issue with CUDA IPC or Metal buffer handles.

Across physical machines, something has to actually move the bytes.

---

# V1 vs eventual fast path

### V1

I'd implement this:

```text
NVIDIA

CUDA device
 ↓
pinned RAM
 ↓
QUIC
 ↓
Mac RAM
 ↓
MLX

```

It's simple, correct, universal and debuggable.

### Later

We detect capabilities:

```text
NVIDIA + RDMA NIC
Apple + TB5/JACCL
NVIDIA + InfiniBand
shared-memory same host
```

and swap transports:

```text
Generic
    QUIC

CUDA ↔ CUDA
    NCCL

MLX ↔ MLX
    MLX distributed

CUDA ↔ remote accelerator
    GPUDirect/RDMA where possible

same machine
    shared memory / zero-copy
```

But all of those sit underneath the same:

```rust
trait TensorTransport {
    async fn send(
        &self,
        tensor: RuntimeTensor,
        dst: NodeId,
    ) -> Result<()>;

    async fn recv(
        &self,
        expected: TensorSpec,
    ) -> Result<RuntimeTensor>;
}
```

So the scheduler doesn't care whether those 10 KB travelled through NCCL, MLX, QUIC or eventually RDMA.

### The key architectural decision

**AetherMesh does not try to create a universal GPU tensor object.**

It creates:

```text
       LogicalTensor
       shape/dtype/layout
             │
      ┌──────┼───────┐
      ▼      ▼       ▼
    CUDA    MLX      CPU
   Tensor   Array   Tensor
```

and the network boundary is:

```text
Runtime Tensor
      ↓
canonical contiguous BF16/F16 bytes
      ↓
network
      ↓
Runtime Tensor
```

That layer is what makes **NVIDIA + Apple + x86 in one inference graph actually practical**.

---

## User

Now that we know that this thing is possible, why the fuck has nobody built it yet? What is the elephant in the room? \

---

## Assistant

The short answer is: **people have built pieces of it, but the universal “throw any NVIDIA/Mac/x86 box into one pool and get efficient inference” version runs into a brutal performance + systems problem.** I’m checking the closest existing systems so I can separate what is genuinely unsolved from what already exists.

---

## Assistant

I found the important correction first: **this has been built partially already.** `llama.cpp` RPC can expose remote CUDA, Metal and CPU devices in one graph and split weights/KV among them; Petals demonstrated heterogeneous internet-scale pipeline inference; exo handles topology-aware multi-device inference. But llama.cpp still labels its RPC backend **proof-of-concept, fragile and insecure**, which is a good clue that the hard part is not “can tensors move?”

---

## Assistant

Yes. There is an elephant in the room, and it’s bigger than the CUDA↔MLX conversion problem.

## The biggest problem: autoregressive decoding is serial

For a dense LLM, generating token `N+1` requires finishing token `N`.

Imagine:

```text
32 GB model

NVIDIA     Apple       x86
layers     layers      layers
0–20       21–40       41–60
   │          │           │
   └──────────┴───────────┘
```

To generate **one token**:

```text
NVIDIA
   ↓
network
   ↓
Apple
   ↓
network
   ↓
x86
   ↓
logits
   ↓
sample token
```

Only **after all of that finishes** can the next token begin:

```text
token 1: NVIDIA → Apple → x86 → sample
token 2: NVIDIA → Apple → x86 → sample
token 3: NVIDIA → Apple → x86 → sample
```

You cannot normally do:

```text
NVIDIA working token 3
Apple working token 2
x86 working token 1
```

for one conversation, because NVIDIA doesn't know token 2 until the entire pipeline finishes token 1.

That's the fundamental dependency.

Multiple independent requests can fill the pipeline, so throughput can improve. But **single-user inter-token latency does not magically become parallel.**

Petals demonstrated exactly this sort of distributed pipeline inference over heterogeneous machines, including over the internet, so the concept absolutely works. citeturn474591search3turn474591search7

---

## Now add one shitty machine

Suppose:

```text
RTX stage       4 ms
Apple stage    11 ms
x86 stage      80 ms
network         4 ms
--------------------
~99 ms / token
```

You're around:

```text
~10 tok/s
```

Remove the x86 compute stage:

```text
RTX      8 ms
Apple   17 ms
network  2 ms
-------------
27 ms

~37 tok/s
```

So adding another computer can make inference **dramatically slower**.

The older exo architecture documentation explicitly acknowledges this behavior: adding less-capable heterogeneous devices can increase aggregate throughput while slowing individual inference latency. citeturn474591search2

That is elephant #1:

> **Compute capacity is additive. Latency isn't.**

A heterogeneous cluster tends toward the characteristics of its slowest critical stage.

---

# And then comes elephant #2: the network

For decode, activation tensors aren't necessarily enormous.

You might send something like:

```text
[1, 1, 8192]
BF16

≈ 16 KB
```

So people initially think:

> 16 KB? Network isn't a problem.

Bandwidth isn't necessarily the issue.

**Latency is.**

If your pipeline has:

```text
NVIDIA → Apple → x86
```

you cross machine boundaries on **every generated token**.

Even:

```text
1 ms
+
1 ms
+
1 ms
```

starts mattering when the actual GPU computation itself takes only a few milliseconds.

This is exactly why current systems increasingly care about RDMA. Current `llama.cpp` RPC supports RDMA and otherwise falls back to TCP; exo specifically advertises Thunderbolt RDMA and topology-aware planning because reducing inter-device latency materially changes whether distributed inference is worthwhile. citeturn664510search0turn474591search0

---

# Elephant #3: NVIDIA → CPU → network → Apple isn't free

Earlier we discussed:

```text
CUDA VRAM
   ↓
cudaMemcpy D2H
   ↓
host RAM
   ↓
network
   ↓
Mac memory
   ↓
MLX
```

That works.

But compare it with two NVIDIA GPUs connected over an appropriate high-speed fabric:

```text
GPU A
 ↓
NCCL / RDMA
 ↓
GPU B
```

The heterogeneous path introduces:

```text
GPU synchronization
device → host copy
serialization
network
deserialization
runtime tensor creation
possibly dtype/layout conversion
```

And then you do that again next token.

So **possible ≠ efficient**.

---

# Elephant #4: "32 GB pooled memory" is kind of a lie

Suppose I give you:

```text
NVIDIA VRAM     8 GB
Mac memory     16 GB
x86 RAM        64 GB
--------------------
Total          88 GB
```

You don't really have an:

```text
88 GB GPU
```

You have:

```text
8 GB very-fast CUDA memory
16 GB Apple unified memory
64 GB relatively slow CPU memory

connected through much slower links
```

It's more like a NUMA machine taken to an extreme.

AetherMesh therefore must understand:

```text
capacity
≠
bandwidth
≠
compute
≠
latency
```

Simply summing RAM is almost meaningless.

---

# Elephant #5: runtime compatibility becomes disgusting

Imagine Gemma layer 32.

CUDA backend might support:

```text
FlashAttention implementation A
Q4_K quantization
fused RMSNorm
CUDA-specific GEMM layouts
```

MLX might support:

```text
MLX attention implementation
different quantization representation
different fused operators
different tensor layouts
```

CPU backend:

```text
GGML kernels
AVX512
yet another quantization layout
```

The mathematical model is the same.

But:

```text
the physical representation isn't.
```

So AetherMesh needs to guarantee:

> "This exact model revision can execute layers 0–20 through CUDA, 21–40 through MLX, and 41–60 through CPU while producing compatible activations."

That's a **model × runtime × quantization × architecture compatibility matrix**.

That becomes enormous.

`llama.cpp` has a significant advantage here because GGML provides one graph abstraction with many backends—CUDA, Metal, CPU, Vulkan, etc.—which is why its RPC system can already expose CUDA, Metal and CPU remotely. citeturn664510search6turn664510search0

If we independently combine:

```text
MLX
+
CUDA runtime A
+
Candle
+
GGML
```

we have much more engineering work.

---

# Elephant #6: KV cache makes failures painful

Imagine:

```text
NVIDIA owns layers 0–20
Mac owns 21–40
x86 owns 41–60
```

After a 40,000-token conversation, each one has accumulated its part of the KV cache.

Then someone's MacBook closes its lid.

😂

Now:

```text
layers 21–40 weights → gone from execution

AND

KV 21–40 → gone
```

You can't simply say:

```text
"okay, move those layers to NVIDIA"
```

because NVIDIA also needs the historical KV state.

Your choices become:

```text
restore KV from replica/cache

or

re-run the entire 40K-token prompt
```

Current production inference systems put enormous effort into KV transfer/offloading precisely because moving/reconstructing state is nontrivial. vLLM, for example, has dedicated connectors for moving KV between prefill and decode instances and multi-tier KV mechanisms. citeturn664510search1turn664510search8

---

# And consumer machines are unreliable

Your data center H100 doesn't:

```text
close its lid
switch Wi-Fi
enter battery saver
thermal throttle because someone starts Chrome
go to sleep
change IP
leave the house
```

Your distributed consumer cluster does.

Dense pipeline inference has an ugly property:

```text
Node A healthy
Node B healthy
Node C DEAD
        ↓
whole pipeline DEAD
```

unless you have replicas/recovery.

---

# So why do datacenters use homogeneous GPUs?

Because this:

```text
H100 ─ NVLink ─ H100
 │               │
 H100 ─ NVLink ─ H100
```

is massively easier to optimize than:

```text
RTX 4090
   │ Ethernet
MacBook M5
   │ Wi-Fi
Ryzen PC
   │
old laptop
```

Same kernels.

Same dtype support.

Same collective libraries.

Predictable performance.

Predictable network.

Predictable failures.

That's why systems like vLLM focus heavily on known accelerator environments and specialized high-speed KV/collective transports. citeturn664510search1turn664510search4

---

# So has somebody actually built AetherMesh already?

**Pieces of it, yes.**

The closest things today are:

- **llama.cpp RPC:** CUDA + Metal + CPU remote devices, automatic memory-proportional splitting, TCP/RDMA. But its own README still describes the RPC backend as proof-of-concept, fragile and insecure. citeturn664510search0
- **exo:** automatic discovery, topology-aware partitioning, MLX distributed, tensor parallelism and Thunderbolt RDMA, presently heavily focused on Apple/MLX clusters. citeturn474591search0
- **Petals:** heterogeneous volunteer/internet pipeline inference; demonstrated that geographically distributed LLM inference can actually work. citeturn474591search3
- **vLLM:** sophisticated production GPU inference, KV movement/offloading and disaggregated serving, but it solves a different, much more controlled deployment problem. citeturn664510search1turn664510search8

Nobody has really nailed:

```text
throw arbitrary compute at it

        ↓

Apple
NVIDIA
AMD
CPU
different RAM
different networks
different storage

        ↓

automatically determine:

runtime
quantization
partitioning
cache placement
parallelism
routing

        ↓

and actually get GOOD performance
```

as a polished general-purpose product.

---

# That is where I would change our AetherMesh thesis slightly

I **wouldn't market it as**:

> Combine every computer and make one giant GPU.

Because physically, that's not what we're getting.

I would define it as:

> **AetherMesh turns heterogeneous compute into an intelligently scheduled inference fabric.**

Meaning sometimes it says:

```text
I see:

RTX 4090
M5 Mac
64GB x86 PC

Best plan:

4090     → compute
M5       → compute
x86      → model cache + prefix KV

Do NOT use x86 for forward pass.
```

Other time:

```text
Model won't otherwise fit.

Use x86 for last 7 layers.

WARNING:
expected 4.8 tok/s instead of 22 tok/s.
```

And maybe:

```text
32 GB BF16 model doesn't fit efficiently.

But Q6 does.

Recommended:
quantize → Q6
use NVIDIA + Apple
leave CPU out

Expected: 17 tok/s
```

**That planner is the product.**

The universal tensor transfer layer is engineering.

The hard research/product problem is deciding:

> **When is adding a resource beneficial, when is it merely necessary to make the model fit, and when should we deliberately ignore it?**

If we solve *that* well, then I think there is something genuinely interesting here.

---

## User

what can we do for this elephant

---

## Assistant

We can’t remove the causal dependency of autoregressive decoding, but we can **design around it** so the slowest heterogeneous node is rarely on the critical path. The strongest levers are speculative decoding, fewer cross-runtime boundaries, prefill/decode separation, and aggressive planner policies that sometimes refuse to use a node.

---

## Assistant

We can attack it. We **cannot remove autoregressive causality for exact normal decoding**, but we can make AetherMesh avoid paying the full distributed penalty for every single token.

The most important idea I'd add is **speculative decoding specifically designed around the distributed pipeline**.

### 1. Stop sending one token through the whole cluster every time

Naive AetherMesh:

```text
token 1 → NVIDIA → Apple → NVIDIA → output
token 2 → NVIDIA → Apple → NVIDIA → output
token 3 → NVIDIA → Apple → NVIDIA → output
token 4 → NVIDIA → Apple → NVIDIA → output
```

That's terrible because every token pays all network RTTs.

Instead, run a small **draft model** somewhere cheap/fast:

```text
Small draft model
        ↓
guesses:

"the"
"quick"
"brown"
"fox"
"jumps"
```

Then send **all 5 candidate tokens through the big distributed model together**:

```text
                5-token verification

NVIDIA stage
     ↓
     ↓ one activation batch
     ↓
Apple stage
     ↓
     ↓ one activation batch
     ↓
NVIDIA stage
     ↓
verify 5 candidates
```

Suppose 4/5 are correct.

Instead of:

```text
4 tokens
=
4 complete distributed round trips
```

we've potentially produced:

```text
4 accepted tokens
≈
1 distributed verification pass
+ cheap drafting
```

This is exactly why speculative decoding works: target models can verify multiple proposed tokens in parallel much more efficiently than producing them one-by-one. llama.cpp already supports this general technique, and vLLM now has parallel drafting implementations such as P-EAGLE. citeturn343743search8turn343743search10

That could be **the killer optimization for AetherMesh**.

---

### 2. There's an even crazier 2026 research direction that is almost exactly our problem

A paper from May 2026 called **Speculative Pipeline Decoding** specifically targets pipeline-parallel LLM decoding.

Normal pipeline:

```text
Token N

Stage A → Stage B → Stage C

then Token N+1
```

Most stages sit idle much of the time.

Their idea is roughly to use speculation so multiple future-token candidates occupy different pipeline stages simultaneously:

```text
time →

Stage A    T1   T2?  T3?  T4?
Stage B         T1   T2?  T3?
Stage C              T1   T2?
```

So instead of:

```text
A working
B idle
C idle

A idle
B working
C idle

A idle
B idle
C working
```

we move toward:

```text
A working on speculative T3
B working on speculative T2
C validating T1
```

The paper explicitly describes this as attempting **zero-bubble speculation using pipeline parallelism**. citeturn343743academia48

I would absolutely research this for AetherMesh.

---

## 3. The scheduler should aggressively keep bad machines OFF the decode path

This is probably even more important.

Say we have:

```text
RTX 4090       24 GB
Mac M5         16 GB
x86            128 GB RAM
old Mac        16 GB
```

It would be stupid to do:

```text
4090 → M5 → x86 → old Mac
```

just because all four machines exist.

Instead:

```text
                 AetherMesh

RTX 4090 ────────┐
                 ├── HOT DECODE ISLAND
Mac M5 ──────────┘


x86 ─────────────── model cache
                     prefix cache
                     cold KV
                     coordinator

old Mac ──────────── draft model
                     background prefill
                     batch work
```

So every resource contributes, but only fast-enough resources touch the latency-critical path.

That's a massive distinction.

---

## 4. Quantize specifically to shrink the critical path

Consider a model that's:

```text
32 GB
```

and therefore requires:

```text
4090 + Mac + x86
```

Full model:

```text
4090 → Mac → CPU
                ↑
             horrible
```

But maybe Q6 brings it to:

```text
~20 GB
```

Now:

```text
4090 only
```

or:

```text
4090 + Mac
```

becomes possible.

Even if Q6 technically does more quantized computation, eliminating:

```text
CPU stage
+
network boundary
```

could make the whole system far faster.

Therefore planner should evaluate:

```text
FP16:
  3 nodes
  estimated 5 tok/s

Q8:
  2 nodes
  estimated 18 tok/s

Q6:
  1 node
  estimated 35 tok/s

Q4:
  1 node
  estimated 45 tok/s
```

subject to your quality constraint.

This means **quantization becomes a scheduling decision**, not merely a model-download choice.

---

# 5. Minimize runtime boundaries

Suppose we have:

```text
NVIDIA A
Apple A
NVIDIA B
Apple B
```

This would be horrible:

```text
CUDA
 ↓ network/conversion
MLX
 ↓
CUDA
 ↓
MLX
```

Three heterogeneous boundaries.

Instead create islands:

```text
NVIDIA A + NVIDIA B
         │
         │ ONE heterogeneous boundary
         ▼
Apple A + Apple B
```

Within NVIDIA:

```text
NCCL
```

Within Apple:

```text
MLX Distributed
```

Only once:

```text
CUDA representation
       ↓
AetherWire
       ↓
MLX representation
```

Planner objective should contain a **very high penalty for heterogeneous boundaries**.

Something like:

\[
Cost =
Compute +
Network +
RuntimeBoundaryPenalty
\]

This means physical layer balancing might intentionally be uneven if it eliminates a runtime crossing.

---

# 6. Separate PREFILL from DECODE

This is another major optimization.

LLM inference really has two workloads:

```text
PROMPT
 ↓
PREFILL

"Read these 20,000 tokens"
```

versus:

```text
GENERATE

token
token
token
token
```

They behave differently.

Prefill likes:

```text
parallel compute
large batches
bandwidth
```

Decode likes:

```text
extremely low latency
fast KV access
low network RTT
```

Current vLLM supports disaggregated prefill precisely so those two phases can use different execution strategies and resources. citeturn343743search0

AetherMesh could do:

```text
                 PREFILL POOL

NVIDIA + Mac + x86 + whatever
           │
           │ produces KV
           ▼
       KV transfer
           │
           ▼

                 DECODE POOL

       fastest 1–2 machines
           │
       token generation
```

The caveat is important:

**the decode nodes still need the model weights necessary for decode.**

So this gets particularly interesting when combined with:

```text
quantization
+
different prefill/decode model representation
```

For example:

```text
Prefill:
BF16
big distributed cluster

Decode:
Q6
2 fast machines
```

Whether the resulting output/quality semantics are acceptable would need careful validation.

---

# 7. Slow machines can run the draft model

This might be one of our best uses for machines that otherwise shouldn't touch the big model.

Imagine:

```text
RTX 4090 + M5
      ↓
32B target model

x86 / old GPU
      ↓
1B draft model
```

The weak machine continuously guesses:

```text
T+1
T+2
T+3
T+4
```

The powerful distributed cluster verifies those guesses.

So instead of the weak machine being:

> dead weight

it becomes:

> **speculation compute**

There is current research around distributed draft/verify pipelines at the edge; DiP-SD, for example, explores devices generating drafts while a more powerful target server verifies them. citeturn343743academia50

That's extremely aligned with AetherMesh.

---

# 8. Use multiple requests to fill pipeline bubbles

Single-user:

```text
Request A

GPU → Mac → GPU
```

has idle stages.

But imagine:

```text
100 requests
```

Then:

```text
time →

GPU1    A1   B1   C1   D1
Mac          A1   B1   C1
GPU2              A1   B1
```

Now the stages remain busy.

So AetherMesh should have two modes:

```text
INTERACTIVE

Optimize:
TTFT
inter-token latency
speculation
minimum number of nodes
```

and:

```text
SERVER

Optimize:
aggregate tok/s
microbatching
pipeline utilization
request concurrency
```

The optimal placement is completely different.

---

# 9. Cache the hell out of everything

Suppose you ask:

```text
"Here is my 100-page codebase..."
```

and then ask 50 questions about it.

We shouldn't prefill that huge common prefix 50 times.

Store:

```text
prefix hash
 ↓
KV blocks
```

Then subsequent requests begin from cached KV.

Slow/large-memory machines are excellent for:

```text
prefix KV store
```

while active sections get promoted onto fast nodes.

vLLM's current disaggregated architecture similarly uses dedicated KV connectors and supports CPU/filesystem/multi-tier KV storage. citeturn343743search0

---

# 10. Failure: asynchronously checkpoint KV

Instead of replicating every KV update synchronously:

```text
GPU
 ↓ every token
x86 backup
```

which destroys latency, do:

```text
HOT KV
GPU/Mac

        ─── asynchronous ──►

WARM KV checkpoint
x86 RAM
```

Maybe every:

```text
64
128
256 tokens
```

depending on policy.

If the Mac disappears at token 10,832 and latest checkpoint is 10,752:

```text
restore KV at 10,752
replay 80 tokens
resume
```

instead of replaying the entire 20K prompt.

So we trade:

```text
small recovery window
```

for:

```text
almost zero steady-state latency overhead
```

---

# What I would make the AetherMesh architecture eventually look like

```text
                         AETHERMESH
                              │
                 ┌────────────┴────────────┐
                 │                         │
             PLANNER                  MODEL ANALYZER
                 │
                 ▼
        RESOURCE CLASSIFICATION
                 │
       ┌─────────┼──────────┬────────────┐
       │         │          │            │
       ▼         ▼          ▼            ▼

    DECODE     PREFILL     DRAFT        STORAGE
    ISLAND     ISLAND      ISLAND       ISLAND

 NVIDIA       GPUs        CPU/GPU        x86
 Apple        Macs        old Mac        SSD
   │
   │
   ├── native NCCL/MLX where possible
   │
   ├── minimum cross-runtime boundaries
   │
   ├── speculative verification
   │
   └── hot KV local
```

So instead of thinking:

> **How do we distribute one model evenly across everything?**

I'd change the problem to:

> **How can every available resource improve some part of inference without slowing the latency-critical path?**

That is much stronger.

### And the one thing I'd bet on hardest

If we want AetherMesh to become genuinely technically interesting rather than just another llama.cpp RPC frontend, I would investigate:

**heterogeneous pipeline placement + speculative pipeline decoding.**

Current speculative decoding already makes multiple-token verification practical, and 2026 research is explicitly exploring keeping pipeline stages busy during single-sequence decoding rather than accepting the traditional pipeline bubble. citeturn343743search3turn343743academia48

That potentially attacks the **actual fundamental weakness** of this project, rather than merely making tensor transport faster.

---

If you want, I can:

- Continue detailing speculative pipeline decoding research
- Describe planner design for heterogeneous resource scheduling
- Explain failure recovery strategies in distributed LLM inference
