---
sprint: sprint-10
status: done
start: 2026-09-17
end: 2026-09-30
owner: us4-core
---

# Sprint 10 — Batching + Speculative Decoding (Apple)

## Objetivo
Continuous batching multi-sessao. Speculative decoding P-EAGLE / EAGLE-3. Draft model loader. Cross-session KV isolation.

## Tasks
- [x] T10.1 — `runtime/scheduler/ContinuousBatcher` (token-level scheduling, fairness)
- [x] T10.2 — `runtime/scheduler/SessionPool` (multi-session state + KV namespace)
- [x] T10.3 — `runtime/speculative/PEagleDecoder` (draft + verify)
- [x] T10.4 — `runtime/speculative/Eagle3Decoder` (n-gram + tree verify)
- [x] T10.5 — Draft model loader (small companion model, shared tokenizer)
- [x] T10.6 — Acceptance rate telemetry

## Test plan
- Unit: batcher fairness; speculative accept/reject correctness; session isolation.
- Regression: adapters single-session verdes.
- E2E: 4 sessoes paralelas geram tokens sem leak; speculative >=1.8x speedup.
- Correctness: tokens identicos a non-speculative path.

## DoD
- Batching + speculative em FULL/BALANCED_PLUS.
- Coverage >=80% em `runtime/scheduler` + `runtime/speculative`.
- ADR-007 Speculative decoding choice.
