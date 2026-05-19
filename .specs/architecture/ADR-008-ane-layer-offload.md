# ADR-008: ANE Layer Offload Heuristic

- Status: Accepted
- Date: 2026-05-18
- Owners: us4-core

## Context

Apple Neural Engine (ANE) becomes usable as a runtime backend on M5+ chips.
The runtime needs an explicit policy for which layers go to ANE and a
graceful fallback contract for chips that do not support it.

## Decision

ANE eligibility is opt-in. The runtime never assumes ANE availability; the
`AneBackend::Probe` routine consumes the `HardwareProbeResult` and reports
one of:

- `available=true`, `fallbackReason="ready"` (M5+ with ANE flag);
- `available=false`, `fallbackReason="chip-too-old"` (everything else).

Layer offload uses a small allow-list: a layer goes to ANE only when its
kind is one of `kAttention`, `kMlp`, `kProjection` **and** `isStatic=true`.
Anything carrying KV state stays on Metal because ANE expects static graphs.

The mixed dispatch coordinator (`BuildMixedDispatchPlan`) flattens the ANE
and Metal layers into an ordered list with the chosen backend per step. The
list is observable through CLI/bench output so fallback events are visible.

Thermal management uses `DecideThermalDowngrade` over a `ThermalReading`.
States `kSerious` and `kCritical` trigger downgrade with stable reason tags
(`serious-thermal`, `critical-thermal`).

## Consequences

- M1-M4 hosts keep Metal as the dense backend; ANE selection silently maps
  to Metal with `chip-too-old` reason on those chips.
- Adapters do not call into CoreML directly; they consume the layer offload
  plan instead.
- The runtime never crashes on hosts without ANE: the contract surface
  works without the toolchain and reports `compile-deferred` until the
  CoreML integration lands.
- Bench evidence includes the per-step backend assignment so we can
  attribute observed speedups to ANE vs Metal correctly.

## Alternatives considered

- Whole-model ANE offload: blocks MoE/dense paths that rely on KV state;
  rejected.
- Heuristic based on layer index only: brittle when adapters reshuffle the
  graph; rejected.
