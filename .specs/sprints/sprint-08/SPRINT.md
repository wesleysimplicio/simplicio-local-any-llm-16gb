---
sprint: sprint-08
status: in_progress
start: 2026-08-20
end: 2026-09-02
owner: us4-core
---

# Sprint 08 - MoE Foundation (Apple)

## Objetivo

MoE: DeepSeek + Kimi adapters. Expert pager (carrega experts on-demand). Top-k
routing. Offload de experts cold pra RAM.

## Estado atual no repo em 2026-05-18

- `runtime/moe/Router` agora executa top-k com softmax e preserva um
  `RouterDecision` com `entropy`, `load_balance`, `selected_mass` e
  `total_experts`.
- `DeepSeekMoEAdapter` e `KimiMoEAdapter` continuam usando a fundacao leve de
  routing/pager, mas agora tambem expoem telemetria MoE agregada no resultado
  nativo (`moe_selected_experts`, `moe_router_entropy`, `moe_load_balance`,
  `moe_selected_mass`).
- `runtime/moe/ExpertPager` agora expõe `load`, `eviction`, `reuse` e
  `residentCount` de forma visível, com `ExpertPagerSnapshot` preservando os
  experts residentes desta chamada.
- Lazy load por expert, shards e corretude externa continuam pendentes neste
  sprint.

## Tasks

- [x] T08.1 - `runtime/moe/Router` (top-k softmax, expert selection, load balance)
- [x] T08.2 - `runtime/moe/ExpertPager` (page experts em unified memory, evict LRU)
- [ ] T08.3 - `runtime/adapters/deepseek/DeepSeekMoEAdapter` (config, shared experts, routed experts)
- [ ] T08.4 - `runtime/adapters/kimi/KimiMoEAdapter`
- [ ] T08.5 - Loader MoE: lazy load por expert (sharded weights)
- [ ] T08.6 - Telemetry: expert hit-rate, eviction count, router entropy

## Test plan

- Unit: router top-k correctness; pager evict re-load; load balance loss.
- Regression: dense adapters + Llama nao quebram.
- E2E: DeepSeek V2/V3 Q4 em M3 Ultra gera 100 tokens em <= 30s.
- Correctness: diff vs HF reference <= 1e-3.

## DoD

- DeepSeek + Kimi em FULL/BALANCED_PLUS.
- Coverage >=80% em `runtime/moe` + adapters MoE.
- ADR-006 MoE expert paging strategy.
