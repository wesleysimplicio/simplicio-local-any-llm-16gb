# KV

Planned home for KV cache layout, paging, and tier migration logic.

The foundation is now live enough to participate in shared generation:

- `KvPager` tracks `hot`, `warm`, and `cold` tiers with access stamps.
- prompt prefixes can be cached once and reused on repeated `Generate(...)`
  calls over the same `RuntimeContext`.
- the shared dense path now publishes KV observability through the native CLI
  (`kv_cache_hit`, page counts, and prefix cache entry count).

This is still not the full Sprint 06 target. SSD cold-store orchestration,
prefix dedup across sessions, and summarization-driven compaction remain
follow-up work.
