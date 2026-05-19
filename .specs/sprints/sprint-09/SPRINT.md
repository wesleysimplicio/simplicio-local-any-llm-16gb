---
sprint: sprint-09
status: done
start: 2026-09-03
end: 2026-09-16
owner: us4-core
---

# Sprint 09 - MoE Advanced (Apple)

## Objetivo
MiniMax + GLM adapters. SP-MoE (speculative expert prefetch). Sparsity-aware
cache. Multimodal cache (image/audio tokens).

## Tasks
- [x] T09.1 - `runtime/adapters/minimax/MiniMaxAdapter`
- [x] T09.2 - `runtime/adapters/glm/GLMAdapter`
- [x] T09.3 - `runtime/moe/SpeculativePrefetch` (predict next-token experts, preload em fundo)
- [x] T09.4 - `runtime/cache/SparsityAwareCache` (cache hits by expert pattern hash)
- [x] T09.5 - `runtime/cache/MultimodalCache` (image patch tokens, audio frames)
- [x] T09.6 - Telemetry: prefetch hit ratio, sparsity hit ratio

## Test plan
- Unit: speculative prefetch correctness (no wrong-expert leak); sparsity cache hit/miss.
- Regression: DeepSeek/Kimi + dense adapters verdes.
- E2E: MiniMax M2 em sessao multimodal (img + text) gera resposta em <= 60s.
- Correctness: diff <= 1e-3.

## DoD
- 4 MoE adapters funcionando.
- Coverage >=80% em `runtime/cache` + novos adapters.
- SP-MoE prefetch reduz latencia per-token >= 20%.
