---
sprint: sprint-11
status: done
start: 2026-10-01
end: 2026-10-14
owner: us4-core
---

# Sprint 11 — ANE M5+ Offload (Apple)

## Objetivo
Apple Neural Engine backend para chips M5+. Offload de dense layers (attention/MLP estaticos) pro ANE. Mixed dispatch Metal+ANE. Thermal/throttle aware.

## Tasks
- [x] T11.1 — `runtime/ane/AneBackend` (CoreML model compile, predict)
- [x] T11.2 — `runtime/ane/LayerOffloader` (chooses which layers go to ANE)
- [ ] T11.3 — Mixed dispatch coordinator (Metal hot path + ANE static layers)
- [ ] T11.4 — `runtime/tuning/ThermalMonitor` (read `IOPMrootDomain`/`powermetrics`, downgrade dispatch)
- [ ] T11.5 — Bench Llama/Qwen em ANE + Metal vs Metal-only
- [ ] T11.6 — Fallback graceful em chips < M5

## Test plan
- Unit: layer offloader picks valid layers; thermal monitor reads signal.
- Regression: chips M1-M4 ainda usam Metal sem regressao.
- E2E: M5+ Llama 8B com ANE offload >= 1.3x tokens/s vs Metal-only; throttle event derruba bem.
- Correctness: ANE diff vs Metal <= 5e-3.

## DoD
- ANE habilitado em FULL para M5+; Metal continua em M1-M4.
- Coverage >=80% em `runtime/ane`.
- ADR-008 ANE layer offload heuristic.
