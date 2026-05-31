# Changelog

All notable changes to **US4 V6 Apple Edition** are recorded here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adopts [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.7] - 2026-05-31

### Changed

- Refresh Playwright and MLX serve dependencies to the latest compatible
  releases used by the local visual smoke flow.
- Bump starter package metadata `0.1.50` -> `0.1.51` and project runtime
  version `0.2.6` -> `0.2.7`.

## [0.2.6] - 2026-05-30

### Added

- Commit `.simplicio/project-map.json` and `.simplicio/precedent-index.json`
  so downstream LLM executions can load the repository map directly.

## [0.2.5] - 2026-05-29

### Added

- `us4-cli serve` now exposes `--chat-backend <mlx|ollama|custom>` and
  `--chat-upstream <url>` as first-class flags, mapping them to
  `US4_SERVE_CHAT_BACKEND` and `US4_SERVE_CHAT_UPSTREAM` before the Python
  sidecar is executed. This keeps the C++ wrapper aligned with the documented
  Ollama/custom-upstream serve modes, so users no longer need to export env
  vars manually for those two common knobs.

## [0.2.4] - 2026-05-27

### Added

- `scripts/openai_serve.py` now forwards two new opt-in environment variables
  straight to the `mlx_lm.server` child process:
  - `US4_SERVE_PROMPT_CACHE_BYTES`: caps the KV/prompt cache (e.g.
    `268435456` = 256 MiB). Stops a long prompt from ballooning resident RAM
    past a safe envelope on memory-constrained boxes. Validated as a positive
    integer at startup; non-numeric values (e.g. `512m`) are logged and
    skipped rather than passed to the upstream as a cryptic argparse error.
  - `US4_SERVE_MLX_EXTRA_ARGS`: shell-style raw argv appended to the
    `mlx_lm.server` command line. Escape hatch for any flag not exposed as
    a first-class env var (e.g. `"--max-tokens 256 --prefill-step-size 512"`).
  Both are honoured only when `US4_SERVE_CHAT_BACKEND=mlx`; silently ignored
  for `ollama` or custom upstreams. Defaults are unchanged (no flag passed
  when the env var is empty), so existing deployments are not affected.

### Security

- `US4_SERVE_MLX_EXTRA_ARGS` now strips any token (and its value, when in
  `--flag VALUE` form) matching `--host`, `--port`, or `--cors*` before the
  tokens are appended to the `mlx_lm.server` argv. Rationale: Python
  argparse honours the last occurrence of a flag, so a user copy-pasting
  the README recipe onto a cloud VM or shared host could silently override
  the hardcoded `--host 127.0.0.1` with `US4_SERVE_MLX_EXTRA_ARGS="--host
  0.0.0.0"` and expose the unauthenticated local LLM endpoint to the
  network. Rejected tokens are logged at ERROR level so misconfigurations
  surface in the serve log instead of becoming a silent exposure. Network
  binding remains fixed to `127.0.0.1` (or `US4_SERVE_HOST` when set, still
  loopback-validated by `ipaddress.ip_address(...).is_loopback`).

### Changed

- `README.md` section 6.1 documents the new env vars in the configuration
  table, adds a worked recipe for running a 7B-class model natively via MLX
  on an M1 8 GB box (3-bit quant + 256 MiB KV cache cap +
  `US4_SERVE_DISABLE_EMBED=1`), and ships a measured benchmark comparing
  this path against the Ollama-proxy path. Result on M1 8 GB: native MLX
  3-bit returns 80 tokens in ~16 s (~5 tok/s) versus ~22 s (~3.5 tok/s)
  through the Ollama proxy — ~43 % faster while staying within the safe
  RAM envelope. Section 6.5 hardware table notes that 7B 3-bit MLX is now
  a viable comfortable target on 8 GB with the KV cache cap. Section 6.6
  troubleshooting matrix grew two rows: 7B OOM-kill mitigation (switch to
  3-bit + cap KV cache) and the `invalid US4_SERVE_MLX_EXTRA_ARGS` shell
  quoting failure.

## [0.2.3] - 2026-05-27

### Changed

- `README.md` section 6.1 now documents the `US4_SERVE_CHAT_BACKEND=ollama`
  mode end-to-end. New content: the `US4_SERVE_CHAT_BACKEND` and
  `US4_SERVE_CHAT_UPSTREAM` env vars added to the configuration table; a
  worked example wiring us4-v6 as a front for an already-running Ollama
  daemon (`ollama serve` + `ollama pull qwen2.5-coder:7b` + us4-v6 with
  `CHAT_BACKEND=ollama`); an ASCII wire diagram showing chat going to
  Ollama on `11434` and embeddings staying in-process via mlx-embeddings;
  a list of the three situations where this mode beats the MLX path
  (Ollama-based model management, single OpenAI-shape facade in front of
  Ollama shared with other tools, machine that cannot host both an MLX
  child and Ollama). Section 6.6 troubleshooting matrix grew two rows
  covering the most likely failures in Ollama mode (`chat upstream
  unreachable: Connection refused`, `model "<id>" not found`). Validated
  with `qwen2.5-coder:7b` on an M1 8 GB box: chat round-trip through
  us4-v6 -> Ollama returned a 78-token FizzBuzz response in ~22 s wall
  (~3.5 tok/s), confirming the proxy path works under tight RAM.

## [0.2.2] - 2026-05-27

### Changed

- Expanded `README.md` section 6 ("Serve OpenAI-Compatible Endpoint") into a
  full local-LLM guide framed as an Ollama-compatible drop-in. New
  subsections cover: Python sidecar path (no C++ build required, with venv
  setup avoiding `externally-managed-environment`), C++ CLI wrapper path,
  smoke-test curl recipes for `/v1/chat/completions` and `/v1/embeddings`,
  pointing arbitrary OpenAI-shape clients at the local endpoint, an Apple
  Silicon hardware sizing table (M1 8 GB to M4 Max 64 GB+), and a
  troubleshooting matrix for the most common first-run failures
  (`mlx-embeddings is not installed`, `externally-managed-environment`,
  `Address already in use`, missing native binary, ignored `--model` flag,
  swap thrashing on undersized RAM). Documents that the sidecar is
  configured via environment variables (`US4_SERVE_*`) and not CLI flags.

## [0.2.1] - 2026-05-27

### Fixed

- `POST /v1/embeddings` returned 500 `embedding failed` with the default
  `embeddinggemma-300m-bf16` model. Root cause: `mlx-embeddings 0.1.0` ships a
  `generate()` helper that calls `model(**inputs)` with key `input_ids`, but
  the `gemma3_text` model class expects `inputs=`. Replaced the helper call in
  `EmbeddingsBackend.embed` with explicit tokenization and a signature-aware
  dispatch (`inputs=` first, fall back to `input_ids=` on `TypeError`) so the
  backend stays model-agnostic across BERT-, Gemma3-, Qwen3-, and ModernBERT-
  style embedders. Smoke: `/v1/embeddings` returns 768-dim vectors for the
  default model; serve E2E suite still 5/5.

## [0.2.0] - 2026-05-27

### Added

- `us4-cli serve` subcommand exposing a local OpenAI-compatible HTTP endpoint
  on Apple Silicon. Routes:
  - `POST /v1/chat/completions` and `POST /v1/completions` proxied to
    `mlx_lm.server` running as a managed child process on `PORT + 1`.
  - `POST /v1/embeddings` served in-process by an MLX embeddings backend.
  - `GET /v1/models` aggregating the active chat and embedding model ids.
  - `GET /health` (alias `/v1/health`) returning liveness and per-backend
    enablement flags.
- Python sidecar `scripts/openai_serve.py` (stdlib `http.server` only, no
  FastAPI/uvicorn) wired by the CLI via `execvp(python3, script)`.
- Dependency manifest `scripts/requirements-serve.txt` pinning `mlx`,
  `mlx-lm`, and `mlx-embeddings` ranges used by the serve sidecar.
- Contract document `.specs/runtime/SERVE-OPENAI.md` covering endpoints,
  request/response shapes, environment knobs, exit codes, and security posture.
- Defaults: chat model `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`,
  embedding model `mlx-community/embeddinggemma-300m-bf16` (overridable via
  `--chat-model` / `--embed-model` or `US4_SERVE_CHAT_MODEL` /
  `US4_SERVE_EMBED_MODEL`).
- Backend opt-out flags `--no-chat` / `--no-embed` (env:
  `US4_SERVE_DISABLE_CHAT` / `US4_SERVE_DISABLE_EMBED`) so CI smoke and partial
  installs do not require downloading both models.
- CLI script-path overrides `US4_SERVE_SCRIPT` and `US4_SERVE_PYTHON` for
  packaged installs and custom interpreter layouts.
- Playwright E2E spec `tests/e2e/us4-cli-serve.spec.ts` covering health,
  models, disabled-backend, and 404 paths with both backends disabled.

### Changed

- Bumped project version `0.1.49` -> `0.2.0` (CMakeLists.txt, propagates to
  `us4::kUs4Version` via `runtime/core/version.h.in`).
- `us4-cli --help` usage block now documents the `serve` subcommand.

### Integration

- `simplicio-cli` (and any OpenAI-compatible client) can target the local
  endpoint with a single env var, e.g.
  `SIMPLICIO_BASE_URL=http://127.0.0.1:8080/v1` plus `OPENAI_API_KEY=anything`.

### Notes

- The serve path is opt-in. The native runtime, fixtures, probe, and existing
  CLI subcommands are unchanged.
- Real chat completions require `pip install -r scripts/requirements-serve.txt`
  and a working MLX install on Apple Silicon.

## [0.1.x] - prior releases

See git history (`git log --oneline`) for pre-0.2.0 increments.

[0.2.0]: https://github.com/wesleysimplicio/us4-v6-simplicio-apple/releases/tag/v0.2.0
