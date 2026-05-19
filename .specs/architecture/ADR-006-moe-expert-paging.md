# ADR-006: MoE Expert Paging Strategy

- Status: Accepted
- Date: 2026-05-18
- Owners: us4-core

## Context

MoE adapters (DeepSeek, Kimi, MiniMax, GLM) need on-demand expert loading
with bounded resident set size. Loading every expert eagerly defeats the
memory savings that MoE architectures are designed for; loading on every
token defeats latency. The runtime needs an explicit paging policy that the
adapter contract relies on.

## Decision

Expert paging uses the `ExpertPager` interface in `runtime/moe/`. The
contract is:

- `Touch(expertId)` records a hit and promotes the expert to the resident
  set. The most-frequently-touched experts win when the resident-set limit
  is exceeded.
- `ResidentCount()` reports the current resident-set size; bench evidence
  pins this to a stable value per generation step.
- Shard descriptors come from `ParseExpertShardManifest`, which produces a
  deterministic list of `ExpertShardDescriptor` entries describing the
  family, model id, expert index, shard index, weight format, and whether
  the expert is routed-only or shared.

Top-k routing emits `RoutingTelemetry` with the chosen experts, an explicit
entropy value, and a load-balance loss that ranges from `0` (uniform) to
`>0` (router collapse). Together with `MoeTelemetrySnapshot`, these signals
are surfaced through adapter results and bench evidence.

Speculative expert prefetch lives in Sprint 09 (`SpeculativePrefetch`) and
consumes the same shard-manifest contract.

## Consequences

- DeepSeek and Kimi adapters consume routed-only shards through the same
  loader contract; future MoE adapters can layer on without re-implementing
  the manifest parser.
- Telemetry fields `expert_hit_rate`, `expert_eviction_count`,
  `router_entropy`, `router_load_balance`, and `prefetch_hit_ratio` are now
  reserved and stable across CLI/bench output.
- The eviction policy reuses `SelectEvictionVictim` from ADR-005 for
  consistency; expert pager and KV pager share the same hybrid hit-count +
  last-touch ranking.

## Alternatives considered

- Eager loading of every expert: bypasses memory savings and breaks MICRO
  modes; rejected.
- Random eviction: trivially cheap but unstable across runs; rejected.
