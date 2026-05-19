# ADR-007: Speculative Decoding Choice

- Status: Accepted
- Date: 2026-05-18
- Owners: us4-core

## Context

Speculative decoding shortens generation latency by using a small draft model
to propose multiple tokens that the target model then verifies in parallel.
Sprint 10 needs to commit to a stable contract so the runtime can ship
batching + speculative decoding together without breaking the existing
single-session adapter API.

## Decision

US4 V6 Apple Edition adopts the P-EAGLE algorithm as the primary speculative
decoder and EAGLE-3 as the n-gram-tree extension. Both decoders share the
same `SpeculativeDraftToken` interface and the same verify window so the
runtime can pick at dispatch time.

Contract rules:

- The verifier accepts a draft token only when `draft_logit <= target_logit +
  1e-3`, otherwise it halts the window. This keeps the final output
  bit-identical to the non-speculative path.
- Draft models load through `DraftModelLoader` which requires a shared
  tokenizer hash; mismatched tokenizers fail loudly instead of producing
  divergent outputs.
- The continuous batcher (`ContinuousBatcher`) enforces fairness across
  sessions using an explicit per-session quantum. Single-session usage
  degrades to a FIFO.
- The session pool (`SessionPool`) guarantees per-session KV namespace
  isolation: namespaces never collide across active sessions.
- Speculative telemetry is exposed via `SpeculativeTelemetry` with stable
  fields `draft_attempts`, `accepted_tokens`, `rejected_tokens`,
  `acceptance_rate`, and `verify_window`.

## Consequences

- Sprint 10 ships a contract surface that the existing dense/MoE adapters
  can opt into without invasive changes.
- Batching and speculative paths land together so cross-session correctness
  stays observable from day one.
- The runtime stays honest: when a draft model is unavailable, the
  speculative path is skipped and telemetry reports zero acceptance.

## Alternatives considered

- Token-level draft only (no n-gram tree): adequate for simple workloads but
  fails to capture multi-token patterns that EAGLE-3 handles.
- Always-on speculation without verify-window: breaks the bit-identical
  guarantee; rejected.
