# Tendril

**Plan and run LLMs across the machines you already have.**

Tendril looks at a model and at your computers and decides how to run it: on
one machine, split across several, or not at all. It explains every decision
in plain language and tells you exactly what to change when something doesn't
fit.

```text
$ tendril plan gemma-2-9b --node m4-air=m4:16 --node m5-air=m5:16 --link thunderbolt --quantize q8_0

✓ Runs across 2 machines (m4-air → m5-air)
  ~10.7 tok/s per conversation · first token in ~4.97 s

  STAGE  MACHINE      RUNS                                MEMORY                          DECODE
  1      m4-air (M4)  embeddings, layers 0–11             ███████░░░░░░░░░  4.6/10.7 GiB   29 ms
  2      m5-air (M5)  layers 12–41, final norm + LM head  █████████████░░░  9.0/10.7 GiB   65 ms
         network      7.06 KiB per hop per token + sampled token back                     143 µs

Why this plan
  • m4-air can't hold it alone: needs 12.4 GiB … but m4-air allows 10.7 GiB
  • 27 of 41 cuts fit. The fastest (after layer 7) is 2.0 ms quicker but would fill m5-air
    to 93% of its budget; this plan keeps more headroom. Use --goal latency to take it.
  • Embeddings are tied to the output head, so both ends hold a copy (930 MiB).
```

## Install

```bash
git clone https://github.com/hetsaraiya/Tendril && cd Tendril

# Apple Silicon (Metal GPU):
cargo install --path crates/tendril-cli --features metal
# NVIDIA (CUDA toolkit installed):
cargo install --path crates/tendril-cli --features cuda
# Anything else (fast CPU kernels, AVX2/NEON):
cargo install --path crates/tendril-cli

tendril doctor        # check this machine
```

Requires Rust 1.82+. Models come from HuggingFace (safetensors); gated models such as
Llama and Gemma need `HF_TOKEN`.

## Run a model on one machine

```bash
tendril run qwen2.5-1.5b                     # downloads, loads, opens a chat in your terminal
tendril run Qwen/Qwen2.5-7B-Instruct -q q8_0 # quantize on load to halve memory
tendril run ~/models/my-model -p "Hello!"    # one-shot answer from a local folder
```

## Run a model across machines

### Friends first, model later

Nobody downloads the whole model: each machine fetches only its own layers.

```bash
tendril serve                                  # 1. you: open an empty pool
tendril join --token XXXX-XXXX-XXXX-XXXX       # 2. each friend: join (same network)
tendril load Qwen/Qwen2.5-14B-Instruct         # 3. you: pick the model
```

`load` fetches only the model's config, tokenizer and tensor headers, plans the split
over the machines present, and then every machine (yours too) downloads just the byte
ranges of its own layers from HuggingFace. `load` must run with the cluster token (it
reads the one `serve` saved, or `--token`). For gated models (Llama, Gemma) every
machine needs its own `HF_TOKEN`; yours is never sent to anyone.

### Model first

On the machine with the model (the **coordinator**):

```text
$ tendril serve Qwen/Qwen2.5-14B-Instruct

Tendril · serving Qwen/Qwen2.5-14B-Instruct
  Web chat   http://192.168.1.20:8080
  API        http://192.168.1.20:8080/v1  (OpenAI-compatible)
  Add more   run this on another machine on your network to add it:
             tendril join --token 7Q2K-9XMP-4HVD-J3FA
             (if your network blocks discovery: tendril join 192.168.1.20:7420 --token 7Q2K-9XMP-4HVD-J3FA)

17:55:13 ! Qwen2.5-14B doesn't fit on the 1 machine here yet: needs 29.4 GiB, m4-air allows 10.7 GiB …
```

On every other machine, paste the join command — no address needed on the same network
(the cluster is found by a fingerprint of its token; the token itself is never broadcast):

```text
$ tendril join --token 7Q2K-9XMP-4HVD-J3FA
· Looking for your cluster on the local network… ✓ found 192.168.1.20 serving Qwen/Qwen2.5-14B-Instruct
17:55:20 ✓ joined as m5-air — serving Qwen/Qwen2.5-14B-Instruct
17:55:21 ↓ receiving weights for layers 22–47 + head: 6.1 GiB/13.9 GiB (44%)
17:56:02 ✓ running layers 22–47 + head (13.9 GiB, Metal) — ready in 41.3 s
```

![Tendril web chat with the live pipeline view](docs/images/web-ui.png)

As machines join, Tendril measures each link, re-plans automatically and starts serving
as soon as the model fits. Then:

- open the **web chat** (conversation + live view of the pipeline, memory, links and throughput),
- chat from a terminal with `tendril chat`,
- point any OpenAI client at `http://<coordinator>:8080/v1` (serve several models at once:
  `tendril serve modelA modelB`, below),
- watch the cluster with `tendril status`.

What happens under the hood:

- **Only the needed weights move.** Each machine receives exactly the tensors of its
  layers (never the whole checkpoint), streamed from the coordinator and cached on disk,
  so a restart reloads instantly.
- **Only activations cross the network.** Each token sends one hidden-state vector per
  machine boundary and a single token id back — a few KB, not the model.
- **Every link is authenticated and encrypted** (Noise protocol, pre-shared cluster
  token). A machine without the token can't join, read activations or inject work.
- **Machines can come and go mid-answer.** If a machine leaves while a response is
  streaming, the request pauses instead of failing: the cluster re-plans with who's left
  (or waits for the machine to come back — its weights are cached, so rejoining takes a
  second), replays the tokens already committed, and the stream continues. No text is
  lost or repeated; on CPU the continuation is bit-identical. Results from the old plan
  are rejected by epoch. Requests give up only after `--recovery-timeout` (180 s).
- **Continuous batching.** When several conversations are active, each machine runs
  whatever work is waiting — decode steps from different conversations and prefill
  chunks — in one pass, reading every weight once per batch instead of once per request.
  Attention and KV stay per conversation, and batched results are identical to
  one-at-a-time results. Tokens no longer wait behind other people's prompts.
- **Prefix caching.** Chat clients re-send the whole conversation every turn. Tendril
  keeps each finished conversation's KV parked on the machines (inside the memory the
  plan already reserved), so the next turn only computes the new tokens — measured 1.9 s
  → 130 ms time-to-first-token at 1K tokens of history. Idle conversations spill to each
  machine's disk (`--kv-disk 8gb`) and come back when needed. Reuse is exact: output is
  identical to recomputing. OpenAI clients see `usage.prompt_tokens_details.cached_tokens`.
- **Speculative decoding.** Every token normally pays for a full trip through all
  machines. Tendril drafts likely next tokens by *prompt lookup* — copying what followed
  the conversation's latest n-gram the last time it appeared (free; great for code
  edits, summaries, RAG and structured output) — and the pipeline verifies them all in
  one pass. Verification uses exact speculative sampling, so the output distribution is
  unchanged and greedy output is identical. It adapts the draft length and switches
  itself off when drafts aren't paying. Measured on an emulated Wi-Fi link: 107 → 260
  tok/s at 74% acceptance (`--no-speculate`, `--draft-tokens N`).
- **Splitting is exact.** Pipeline execution is bit-identical to running the model on one
  machine (`tendril verify <model>` checks this for any model).

Useful flags: `--context 32k`, `--concurrency 8`, `--quantize q8_0|q4_k`,
`--goal latency|throughput|memory`, `--max-memory 8gb` (cap what Tendril may use on a
machine, on `serve` or `join`), `--min-machines 2` (force a split, e.g. to measure its
cost), `--no-local` (coordinate only), `--kv-disk 16gb` / `--no-prefix-cache`.

### Several models on the same machines

Give `serve` more than one model and they share the machines:

```text
$ tendril serve Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct

Tendril · serving Qwen/Qwen2.5-7B-Instruct + Qwen/Qwen2.5-Coder-1.5B-Instruct
17:02:11 ! Sharing 1 machine(s): Qwen2.5-7B-Instruct → studio (17.2 GiB) · Qwen2.5-Coder-1.5B-Instruct waits
         (fits on its own, but not next to Qwen2.5-7B-Instruct — add a machine (`tendril join`) or serve fewer models)
17:02:40 ✓ laptop joined — Apple M4 · Metal backend · 10.7 GiB for models
17:02:41 · Sharing 2 machine(s): Qwen2.5-7B-Instruct → studio (17.2 GiB) · Qwen2.5-Coder-1.5B-Instruct → laptop (4.1 GiB)
```

- **One join serves them all.** A joining machine opens a session per model and can host
  a slice of several models at once.
- **Tendril divides every machine's memory between the models.** It tries each placement
  order and keeps the one that places the most models (listed order breaks ties: list your
  main model first), then the fastest. Each model then plans inside its own share of
  memory, like it would on a smaller machine.
- **Stable.** A running model moves only when the new placement is ≥25% faster. When
  shares change, a model gives memory back before another model loads into it, and its
  in-flight requests resume on the new placement (see recovery above).
- **The API routes by `model`.** Tendril accepts the exact id, any case, the name without
  the org (`Qwen2.5-Coder-1.5B-Instruct`) or a unique part of the name (`coder`). If you
  leave `model` out, you get the first model. An unknown name gets a 404 that lists the
  served models. `/v1/models` lists every model with its state.
- **The web chat has a model menu.** The cluster panel shows each model's state and share,
  one conversation can hop between models, and each reply is labelled with the model
  that wrote it. `tendril chat -m coder` (or `/model coder` inside the chat) and
  `tendril status` work the same way.

![Two models sharing two machines](docs/images/pool-web-ui.png)

### Measure it

```bash
tendril bench                       # against a running `tendril serve`
tendril bench qwen2.5-1.5b          # or load a model in-process and measure it
tendril bench --concurrency 1,2,4,8 --prompt-tokens 128,2048 --output-tokens 256 --runs 5
tendril bench --turns 5             # add a multi-turn chat: watch the prefix cache work
```

`tendril bench` runs a fixed protocol — warm-up, prompt-length × concurrency sweep,
repeated runs, end-of-sequence ignored so every request generates the same length — and
reports time-to-first-token and inter-token latency percentiles, per-request and total
throughput, the planner's prediction next to the measurement, and **where each token's
time goes** (per-stage compute, queueing, network), from timings every stage stamps onto
the tokens it produces. It writes a self-contained HTML report with charts.

**The planner learns.** While serving, the coordinator compares each machine's measured
compute time with the plan's prediction and stores the ratio
(`~/.config/tendril/calibration.json`); the next plan uses how fast machines really are.
`tendril node --probe` measures memory bandwidth and is reused for two weeks — by `plan`,
`serve`, and by machines when they `join`.

### OpenAI-compatible API

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a haiku about tendrils"}],
  "stream": true
}'
```

`/v1/chat/completions` and `/v1/completions` support streaming, `temperature`, `top_p`,
`top_k`, `min_p`, `seed`, `stop`, `max_tokens`, presence/frequency/repetition penalties
and `stream_options.include_usage` (plus `ignore_eos` for benchmarking); `POST /tokenize`
counts tokens. Responses include a `tendril` object with
time-to-first-token and decode speed. `/api/status` exposes the cluster as JSON.

### Supported models

Architectures: **Llama** (1–3.3, incl. Llama 3 RoPE scaling), **Mistral**, **Qwen 2 / 2.5**,
**Qwen 3**, **Gemma 1 / 2 / 3** (sliding-window attention, soft-capping), **Phi-3**.
Every architecture is verified against HuggingFace `transformers` (max relative logit
error ~1e-6 in f32; see `tools/make_test_models.py`). Weights: bf16/f16/f32 safetensors,
optionally quantized on load to q8_0, q6_k or q4_k.

## Planning without running

## How the planner works

1. **Inspect** the model: layer count, attention layout (including sliding-window layers
   that cap KV growth), tied embeddings, and — when tensor headers are available — the
   exact bytes of every component.
2. **Budget** each machine honestly. On Apple Silicon the GPU may only wire about ⅔ of RAM
   by default (Tendril tells you how to raise it). A plan fits only when
   `weights + KV cache + activation scratch + transport buffers + runtime + safety margin`
   fits every machine, including load-time peaks. Two 16 GB Macs are not a 32 GB GPU.
3. **Search** every ordered subset of machines and, for each, solve an exact dynamic
   program over all contiguous layer cuts (millions of cuts in milliseconds). A machine
   is left out when including it would slow things down — and Tendril says so.
4. **Predict** decode speed (memory-bandwidth bound: each token streams the stage's
   weights), chunked, pipelined prefill, and the per-token network hop plus the sampled
   token's trip back.
5. **Rank** by goal and **advise**: shortest context that fits, quantizations that fit
   (with quality notes), macOS GPU limits, network upgrades, how much memory to add.

Predictions come from hardware specs; they are estimates, clearly labelled as such.

## Project docs

- [Architecture: how the pieces fit and what crosses the network](docs/architecture.md)
