---
sprint: sprint-06
status: todo
start: 2026-07-23
end: 2026-08-05
owner: us4-core
---

# Sprint 06 — KV Memory Architecture (Apple)

## Objetivo
Pager hot-cold de KV cache (unified memory hot, RAM warm, SSD cold). Prefix cache. KV summarization. Eviction policies.

## Estado atual no repo em 2026-05-14
- `runtime/kv/KvPager` e `runtime/kv/PrefixCache` ja existem com cobertura unitaria de contrato.
- A fundacao atual e leve: nao ha `SsdColdStore`, summarizer nem orquestracao completa por mode/tier.
- A geracao dos adapters ainda nao depende de um KV pager completo; este sprint segue como a entrega alvo para esse subsistema.

## Tasks
- [ ] T06.1 — `runtime/kv/KvPager` (page table, LRU, hot/cold tiers)
- [ ] T06.2 — `runtime/kv/PrefixCache` (shared prefix by hash, ref-count)
- [ ] T06.3 — `runtime/kv/SsdColdStore` (mmap, async flush)
- [ ] T06.4 — `runtime/kv/Summarizer` (compress old tokens to summary vector)
- [ ] T06.5 — Eviction policy: LRU + frequency hybrid, cost-aware
- [ ] T06.6 — Adapter hooks: append, lookup, evict, summarize
- [ ] T06.7 — Telemetry: hit-rate hot/warm/cold + summarize ratio

## Test plan
- Unit: pager hit/miss; prefix cache dedup; SSD flush/restore round-trip.
- Regression: adapters Qwen/Gemma/BitNet/Ternary nao quebram.
- E2E: prompt longo (8k tokens) com prefix repetido -> hit-rate hot >= 80%.
- Correctness: KV restaurado de SSD da resultado identico ao mantido em hot.

## DoD
- KV tiering automatico por mode (FULL=hot only, DEGRADED=hot+warm, MICRO=hot+warm+cold+summary).
- Coverage >=80% em `runtime/kv`.
- ADR-005 KV cache architecture.
