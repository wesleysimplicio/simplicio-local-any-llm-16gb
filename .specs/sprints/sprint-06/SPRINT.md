---
sprint: sprint-06
status: done
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
- O dense path compartilhado agora ja evita recompute completo do prompt em geracoes repetidas no mesmo `RuntimeContext`: `DenseAdapterBase::Generate` reutiliza prefixos via `KvPager`/`PrefixCache` e expõe telemetria de hit/page-count no resultado nativo.
- Em `MICRO` e abaixo, esse mesmo path agora faz um primeiro flush para `SsdColdStore`, resume prefixo antigo via `Summarizer`, e mantém so o resumo + janela recente na working set.
- O gap restante de S06 segue sendo tiering mais profundo, restore/evict mais inteligente e hooks dedicados em Llama/MoE.

## Tasks

- [x] T06.1 — `runtime/kv/KvPager` (page table, LRU, hot/cold tiers)
- [x] T06.2 — `runtime/kv/PrefixCache` (shared prefix by hash, ref-count)
- [x] T06.3 — `runtime/kv/SsdColdStore` (mmap, async flush)
- [x] T06.4 — `runtime/kv/Summarizer` (compress old tokens to summary vector)
- [x] T06.5 — Eviction policy: LRU + frequency hybrid, cost-aware
- [x] T06.6 — Adapter hooks: append, lookup, evict, summarize
- [x] T06.7 — Telemetry: hit-rate hot/warm/cold + summarize ratio

## Test plan

- Unit: pager hit/miss; prefix cache dedup; SSD flush/restore round-trip.
- Regression: adapters Qwen/Gemma/BitNet/Ternary nao quebram.
- E2E: prompt longo (8k tokens) com prefix repetido -> hit-rate hot >= 80%.
- Correctness: KV restaurado de SSD da resultado identico ao mantido em hot.

## DoD

- KV tiering automatico por mode (FULL=hot only, DEGRADED=hot+warm, MICRO=hot+warm+cold+summary).
- Coverage >=80% em `runtime/kv`.
- ADR-005 KV cache architecture.
