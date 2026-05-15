# Runtime Scaffold

This directory hosts the early C++ runtime scaffold for US4 V6 Apple Edition.

It is no longer just a planned placeholder: the repo already contains a
buildable skeleton for Sprint 01, while the real inference runtime is still
ahead of us.

## What exists now

- root `CMakeLists.txt` configures the runtime build
- `runtime/CMakeLists.txt` builds `us4_runtime_core`
- `apps/CMakeLists.txt` builds the `us4-cli` smoke executable
- `core/` contains compileable contracts for hardware probe, runtime mode,
  runtime context, and backend selection
- `adapters/` contains a native registry plus scaffold adapters for Qwen,
  Gemma, BitNet, ternary, Llama, DeepSeek MoE, and Kimi MoE
- `tests/fixtures/models/llama-3.1-8b/` now carries both a fixture manifest
  directory and a toy GGUF payload so unit/E2E coverage can exercise the Llama
  loader contract before real weights land
- `metal/` now contains a cross-platform command queue skeleton used by native
  contracts
- `mlx/` now contains a cross-platform bridge skeleton used by native
  contracts
- `kv/` contains small `KvPager` and `PrefixCache` foundations used by unit
  coverage
- repeated prompt generation in the same `RuntimeContext` now reuses prompt KV
  rows instead of rebuilding them from scratch every token step
- `moe/` contains small `Router` and `ExpertPager` foundations used by MoE
  adapter scaffolding
- `telemetry/` contains minimal sink/types placeholders used by smoke tests
- `benchmarks/dense_baseline.cpp` now exercises the scalar dense baseline path
- `tests/unit/` contains smoke coverage plus backend selection, KV, and MoE
  contract tests

## What the scaffold already proves

- the runtime tree layout is real
- build entrypoints are explicit
- `us4-cli` already exposes `--version`, `--probe`, `--mode <value>`, and a
  first `run --model ... --prompt ...` scalar path
- `us4-cli list-models` exposes the native adapter registry
- `run --model-path ...` can load fixture manifests and detect GGUF /
  Safetensors file types without external libraries
- the native CLI contract can already resolve the Llama fixture directory or
  toy GGUF path and surface backend/fallback telemetry without pretending that
  real Llama weights are loaded yet
- backend selection and fallback are explicit in CLI output
- the backend contract already accepts `scalar`, `neon`, `mlx`, `metal`, and
  `ane`, with automatic fallback when a requested path is unavailable
- auto selection now emits explicit reasons such as `auto-metal`, `auto-mlx`,
  `auto-neon`, and `auto-scalar`, which makes mode-driven routing easier to
  inspect in CLI and tests
- `run --json` now also surfaces scaffold observability such as
  `shared_allocations`, `metal_dispatches`, `mlx_operation_count`,
  `metal_device`, and `metal_queue_label`
- `RuntimeContext` now exposes acceleration services for Metal and MLX even
  before the real Apple-only backend code lands
- the native registry already exposes dense, ternary, llama, and MoE-family
  adapters; Llama now owns a dedicated scalar/NEON contract path for prompt KV,
  RoPE-shaped rows, and GQA-flavored generation while Metal/MLX/ANE still use
  the shared scaffold path
- hardware probe and mode selection contracts compile and run
- KV and MoE directories now contain contract-grade foundations, not just empty
  placeholders
- dense adapters can generate deterministic scalar tokens for fixture-grade
  validation

## What is still missing

- full tensor/view ownership model and execution graph
- real weight loading from GGUF / Safetensors payloads
- real MLX bridge and Metal kernels used by generation
- NEON / Accelerate hot paths used by generation
- backend-specific execution beyond the selection and fallback contract
- production KV tiering, SSD cold storage, and summarization flows
- cross-context cold-store and summarization reuse for the dedicated Llama path
- dedicated KV reuse for MoE paths beyond the shared scaffold
- production MoE routing, expert lazy loading, and expert telemetry
- production-capable `run`, `serve`, `bench`, and `tune` CLI flows
- correctness fixtures and backend regression coverage

## Directory intent

| Path                                           | Intent today                                      | Evolves into                                            |
| ---------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------- |
| `core/`                                        | stable contracts and orchestration skeleton       | runtime orchestration, selection, and shared primitives |
| `adapters/`                                    | native registry and scaffold family adapters      | dense, MoE, and low-memory adapters                     |
| `mlx/`                                         | reserved primary backend surface                  | MLX graph/build/eval integration                        |
| `metal/`                                       | reserved accelerated backend surface              | measured hot kernels only                               |
| `neon/`                                        | reserved CPU fallback surface                     | scalar/NEON low-memory and safety paths                 |
| `ane/`                                         | reserved opt-in backend surface                   | validated M5+ offload paths                             |
| `kv/`                                          | contract-grade pager and prefix-cache foundation  | KV lifecycle, tiering, and reuse                        |
| `moe/`                                         | contract-grade router and expert-pager foundation | routed experts and MoE scheduling                       |
| `memory/`, `cache/`, `speculative/`, `tuning/` | roadmap-aligned placeholders                      | runtime subsystems landed by later sprints              |
| `telemetry/`                                   | smoke-level instrumentation contract              | structured runtime metrics and fallback observability   |
| `benchmarks/`                                  | baseline scaffold harness                         | correctness and throughput evidence                     |

## Build entrypoints

From repo root:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/us4-cli --version
./build/us4-cli --probe
./build/us4-cli --mode auto
./build/us4-cli list-models
./build/us4-cli run --model qwen-0.5b --prompt "hi"
./build/us4-cli run --model qwen-0.5b --model-path tests/fixtures/models/qwen-0.5b/model.us4manifest --prompt "hi"
./build/us4-cli run --model llama-3.1-8b --backend metal --prompt "hello"
```

These commands validate the scaffold and CLI contract. They do not validate
real inference yet.

Requesting `--backend metal`, `--backend mlx`, or `--backend neon` currently
validates selection and reporting behavior; it does not prove that generation is
already running on those accelerated paths.

## Llama vertical evidence today

The repo now keeps an explicit pre-vertical contract for Llama in tests and
fixtures:

- `tests/unit/adapter_generation_contract_test.cpp` covers directory-manifest
  loading, default prompt fallback, GGUF routing, requested-backend fallback
  telemetry, and the dedicated Llama KV reuse boundary for the fixture assets,
  including seed-scoped cache partitioning inside a shared `RuntimeContext`;
- `tests/unit/model_asset_contract_test.cpp` and
  `tests/unit/runtime_contract_runner.cpp` also keep the future
  `ResolveLlamaConfig` seam visible by checking fixture metadata hydration plus
  safe normalization for invalid GQA and RoPE fields before the real Sprint 07
  loader lands;
- `tests/e2e/us4-cli.spec.ts` covers native CLI execution against both the
  `llama-3.1-8b/` manifest directory and `toy-llama.gguf`, including host-aware
  assertions for `metal` request behavior;
- these checks are evidence that adapter selection, asset detection, and
  observability stay intact while the real Sprint 07 forward path is still
  under construction.

## Transition rule

During the starter-to-runtime transition, this tree must stay honest about the
current repo state:

- document what already builds and runs;
- mark placeholders as placeholders;
- land real runtime behavior inside the existing ownership boundaries from
  `PATTERNS.md`;
- avoid treating registry presence or backend selection as proof of
  production-grade family/back-end execution;
- avoid claiming MLX, Metal, NEON, or adapter support before the code and tests
  exist.

See [STARTER-TO-RUNTIME.md](STARTER-TO-RUNTIME.md) for the short migration map.
