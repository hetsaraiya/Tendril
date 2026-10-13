# Tendril architecture

Tendril runs no models itself. It finds machines, measures them, plans where a model goes,
moves files, secures the network, starts the inference engine the user picked on the right
machines, and serves one OpenAI-compatible API in front of all of them.

## Crates

| Crate | What it does |
|---|---|
| `tendril-core` | Hardware detection and measurement, model inspection (GGUF, safetensors, HF configs) and the planner: memory per machine, layer splits, predicted speed. No I/O beyond reading headers. |
| `tendril-engines` | The `Engine` trait and one type per engine (llama.cpp, vLLM, SGLang, Ollama, MLX-LM, exo, ComfyUI, sd.cpp, whisper.cpp): detection, model resolution, launch recipes, API translation. |
| `tendril-cluster` | Encrypted links and tunnels, the agent on every machine, the pool on the server, downloads, LAN discovery, the HTTP API and dashboard. |
| `tendril-cli` | The `tendril` command. |

## The engine trait

Each engine is a class that decides its own behaviour (`crates/tendril-engines/src/engine.rs`):

- `info()` — name, tasks (chat / image generation / transcription), input and output
  modalities, how it spans machines (`Single`, `Layers`, `Homogeneous`, `EngineManaged`),
  how its traffic between machines is protected (`Security`), requirements with install
  commands per OS.
- `locate()` / `detect()` — run on every machine: is it installed, which version, which
  accelerators, what is missing and how to install it. Shown per machine in the dashboard.
- `resolve()` — turn what the user typed into a `ResolvedModel`: files to download, a memory
  model for the planner (real layers for GGUF/HF models; a one-block spec for image, audio
  and registry models), whether it takes images.
- `arrange()` — adjust Tendril's placement to how the engine really lays the model out, so
  the plan shown is the plan that runs. llama.cpp puts RPC devices first and the serving
  machine's GPU last (or its CPU layers first), and gets exact `--tensor-split` values.
- `roles()` — the processes to start: program, arguments with placeholders (`{port}`,
  `{file:N}`, `{peer:ROLE}`, …), environment, readiness check, ordering (`after`,
  `after_started`).
- `prepare()` — work after the process is up (Ollama pulls the model, exo places it).
- `api()`, `images()`, `transcribe()` — OpenAI-compatible engines are proxied as they are;
  ComfyUI and whisper.cpp are translated.

`Security` is the per-engine answer to "who encrypts the traffic between machines":

| Kind | Meaning | Engines |
|---|---|---|
| `LocalOnly` | one machine per model; only the API leaves it, through Tendril | Ollama, MLX-LM, ComfyUI, sd.cpp, whisper.cpp |
| `TendrilTunnel` | plain TCP on known ports: bound to localhost, carried by Tendril | llama.cpp |
| `EngineEncrypted` | the engine encrypts and authenticates itself | (none yet) |
| `TrustedNetworkOnly` | dynamic ports Tendril can't wrap: needs the user's consent | vLLM, SGLang, exo |

## Life of a model

1. **Machines join.** `tendril serve` starts the pool and an agent for this machine;
   `tendril join` starts an agent elsewhere. Each agent sends its hardware profile and its
   engine detections.
2. **Plan.** `POST /api/plan`: the engine resolves the model; machines without the engine are
   set aside with what they miss; memory other models were promised is subtracted; the core
   planner splits the layers (one machine for `Single` engines; every machine for
   `EngineManaged`); `arrange()` and `roles()` produce the final placement and commands. The
   preview carries the pool version.
3. **Confirm.** `POST /api/models` repeats the plan and refuses if the pool changed
   (`expect_version`) or if the split needs a trusted network the user didn't confirm.
4. **Run.** The pool sends each role to its machine in dependency order. The agent downloads
   the role's files (parallel ranges, resumable), picks free ports, opens tunnel ends for the
   roles it talks to, expands placeholders, starts the process in its own process group,
   waits for readiness, runs `prepare()`, and reports every step (shown live in the dashboard).
5. **Serve.** When the API role is ready, the pool opens a local forwarder to it (direct on
   the server's machine, a tunnel otherwise) and the HTTP layer proxies requests there.
6. **Stop.** Unload, failure of any role, or its machine leaving stops every process of the
   deployment (SIGTERM, then SIGKILL, to the whole process group) and frees its memory.

## Tunnels

Every agent runs a tunnel listener. A tunnel is a Noise-encrypted connection that names a
target `deployment/role`; the listener forwards it to that role's localhost port only if it
runs the role, and refuses everything else. The other side is a `Forwarder`: a 127.0.0.1
listener whose connections each open a tunnel. llama-server's `--rpc` list and the pool's API
route both point at forwarders, so engines only ever talk to localhost.

## Security

- One cluster token (generated on first `serve`, stored 0600). Every TCP connection performs
  a Noise `NNpsk0_25519_ChaChaPoly_BLAKE2s` handshake keyed by the token; frames are bounded
  before allocation.
- Engines bind 127.0.0.1. Only `TrustedNetworkOnly` engines spanning machines open LAN ports,
  and only after explicit consent; each such deployment gets its own namespace derived from
  the token (exo) — isolation between honest clusters, not authentication.
- A machine's HuggingFace token is sent only to its configured HuggingFace endpoint.
- Control endpoints need the cluster token; the inference API is open to the network the
  server listens on (bind `--host 127.0.0.1` to keep it local).
- Machines in a cluster see what they process. Encryption protects the network, not a
  participant: only join machines you trust.

## Discovery

The server broadcasts a beacon with a fingerprint of the token (never the token) so
`tendril join --token T` finds it without an address on the same network.

## Testing

- Unit tests per crate (tensor-split math checked against llama.cpp's own assignment rule,
  placeholder expansion, tunnels, downloads, planner).
- `tests/pool.rs` uses a stand-in `exo` to check the trusted-network gate and that unloading
  stops every process.
- `tests/engines.rs` runs real llama.cpp: a tiny GGUF (`tools/make_tiny_gguf.py`) on one
  machine, then split across two agents with RPC through the tunnels; the answers must match.
  CI builds llama.cpp with RPC for it.
