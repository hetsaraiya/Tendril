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
cargo install --path crates/tendril-cli     # installs the `tendril` binary
tendril doctor                               # check this machine
```

Requires Rust 1.80+. Works on macOS (Apple Silicon), Linux and Windows.

## Five-minute tour

```bash
# Can I run it? A matrix of quantization × context on this machine.
tendril fit qwen2.5-14b

# Plan for machines you have (or are thinking of buying).
tendril plan llama-3.3-70b --node studio=m2-ultra:192
tendril plan gemma-2-9b --node air=m4:16 --node mini=m5:16 --link thunderbolt --explain

# Any HuggingFace repo: only config.json and tensor headers are fetched (a few KB),
# never the weights.
tendril inspect Qwen/Qwen2.5-7B-Instruct
tendril plan mistralai/Mistral-Nemo-Instruct-2407 --context 32k

# Local folders and GGUF files work too.
tendril plan ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Describe a whole lab in a file.
tendril plan llama-3.3-70b --cluster examples/mixed-lab.toml --quantize q4_k
```

## Commands

| Command | What it does |
|---|---|
| `tendril plan <model>` | Chooses machines, layer split and order; predicts speed and memory; explains the choice. `--explain` shows every cut, the memory breakdown and rejected alternatives. `--json` for scripts. |
| `tendril fit <model>` | "Can I run it?" matrix across representations (bf16/q8/q6/q4) and context lengths. |
| `tendril inspect <model>` | Architecture, where the bytes are, KV cache per context, size per quantization. |
| `tendril node [--probe]` | This machine as the planner sees it; `--probe` measures memory bandwidth. |
| `tendril doctor` | Checks accelerator, memory, macOS GPU limit, power, disk and HuggingFace access, with fixes. |
| `tendril models` | Models known offline (Llama, Qwen, Gemma, Mistral, Phi, DeepSeek-R1 distills, …). |
| `tendril hardware` | Hardware presets for `--node` (M1–M5 families, RTX 30/40/50, A100/H100, CPUs). |

Common flags: `--context 32k`, `--concurrency 4`, `--goal balanced|latency|throughput|memory`,
`--quantize q8_0` (only ever applied when you ask), `--link thunderbolt|10gbe|gbe|wifi`,
`--with-local` (add this machine to `--node` machines), `--offline`.

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

- [What Tendril is and its main challenges](docs/project-and-challenges.md)
- [Approaches and experiments to try first](docs/approaches-and-experiments.md)
- [Original imported project context](docs/project-context.md)
- [Full readable source conversation](docs/source-conversation.md)
