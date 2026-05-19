# Tuning

Hardware-aware tuning surfaces live here. The current slice provides
`ThermalMonitor`, a probe-derived thermal-pressure contract used by
`RuntimeContext` to cap requested modes before backend selection.

Current downgrade semantics:

- `nominal`: keep the requested mode unchanged.
- `elevated`: cap at `BALANCED_PLUS`.
- `critical`: cap at `MICRO_PLUS`.
- `unavailable`: preserve the requested mode and surface `thermal-unavailable`.

This is intentionally observable through `GenerationResult` and `us4-cli run`
as `thermal_pressure_level`, `thermal_reason`, and `thermal_downgraded`.
