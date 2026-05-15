# ADR-003: Backend Selection Strategy

- Status: Accepted
- Date: 2026-05-14
- Owners: us4-core

## Context

After the scalar baseline landed, the runtime needed a stable backend-selection contract before true MLX, Metal, and NEON implementations were all available on every host.

The project already defines:

- MLX as the preferred Apple Silicon path
- Metal as a measured acceleration path
- NEON/scalar as the fallback path
- correctness-first behavior when an accelerated path is unavailable

We also need the CLI and adapters to expose which backend was selected, and whether a fallback happened.

## Decision

We introduce a runtime-level backend selector with these rules:

1. A user may request `scalar`, `neon`, `mlx`, `metal`, or `ane`.
2. If the requested backend is unavailable for the current host, mode, or adapter, the runtime falls back automatically.
3. Automatic selection prefers:
   - `metal`
   - then `mlx`
   - then `neon`
   - then `scalar`
4. The selected backend and the fallback reason must be observable in the generation result and CLI JSON output.
5. Until host-specific accelerated code is fully implemented, unsupported paths remain valid requests that degrade honestly to a safe path.

## Consequences

- The runtime can evolve accelerated backends incrementally without blocking CLI/product flows.
- Tests can assert observable fallback semantics even on non-Apple hosts.
- Adapter families can declare backend capabilities independently from host detection.
- This decision does not claim that MLX/Metal/ANE are implemented everywhere today; it only fixes the selection contract and fallback behavior.
