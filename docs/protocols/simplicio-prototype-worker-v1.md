# `simplicio.prototype-worker/v1`

The local worker consumes the governed backend contract from #142. It is not a
coordinator and has no workspace/tool effect authority.

Supported artifact contracts are `wireframe`, `schema`, `model`,
`failing-test`, `plan` and `prompt-candidate`. Candidates must be JSON-safe,
fit the selected schema and meet the configured quality floor. Invalid or
low-confidence output becomes `escalate_remote`; it is never promoted as a
successful prototype.

The worker is offline-only and does not log prompts. `generate` without a
Runtime inference lease returns `runtime-inference-lease-required`, preserving
the paused local-model policy. Judge execution is rejected unless policy
explicitly permits a distinct judge model.

Use `us4-cli prototype doctor --json` for the capability manifest. The
`critic`, `judge` and `summarize` commands validate supplied candidate JSON
without loading a model. Actual inference remains behind Runtime admission,
budget, lease and fencing.
