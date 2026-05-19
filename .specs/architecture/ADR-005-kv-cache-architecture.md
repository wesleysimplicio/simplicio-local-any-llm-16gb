# ADR-005: KV Cache Architecture

- Status: Accepted
- Date: 2026-05-18
- Owners: us4-core

## Context

Local LLM inference on Apple Silicon stresses memory hierarchies in ways that
single-tier KV caches cannot honour. Sprint 06 needs a stable contract for KV
tiering, prefix reuse, summarization, and eviction so adapters and benchmarks
can rely on consistent behaviour.

## Decision

The KV cache is layered into four tiers:

- **hot**: unified memory, indexed by `KvPager` for fast lookup.
- **warm**: demoted hot pages with no eviction yet.
- **cold**: SSD-backed via `SsdColdStore`; restored lazily.
- **summary**: compacted state produced by `Summarizer` for long contexts.

Prefix reuse uses `PrefixCache` for ref-counted shared prefixes. Eviction is
performed by `SelectEvictionVictim` in `eviction_policy.{h,cpp}` with a hybrid
hit-count + last-touch ranking. The policy is deterministic and reports the
`reason` for each decision so telemetry can attribute eviction events.

KV tiering is mode-driven:

| Mode | Tiers active |
|---|---|
| FULL | hot only |
| BALANCED_PLUS | hot |
| DEGRADED | hot + warm |
| ULTRA_LOW | hot + warm |
| MICRO | hot + warm + cold + summary |
| MICRO_PLUS | hot + warm + cold + summary |
| NANO | hot + warm + cold + summary |

## Consequences

- Adapters share lifecycle hooks via `DenseAdapterBase` and consume the same
  `KvPager` instance through `RuntimeContext`.
- SSD cold-store keys must be Windows-safe and Apple-safe (no `:`, no spaces),
  which is enforced inside `SsdColdStore::Flush/Restore`.
- Summarizer outputs are deterministic for a fixed prompt prefix; tests in
  `tests/unit/kv_contract_test.cpp` cover the contract.
- Telemetry fields (hit_hot, hit_warm, hit_cold, summarize_ratio,
  evict_count, prefix_cache_dedup_count) are exposed via the adapter result
  and the bench evidence.
- Cross-context cold-store reuse is allowed; restored pages keep their stored
  values intact.

## Alternatives considered

- Single-tier hot-only KV cache: too aggressive on memory under MICRO mode.
- LRU-only eviction: loses signal when access counts diverge; the hybrid
  policy still degrades to LRU when frequency ties happen.
