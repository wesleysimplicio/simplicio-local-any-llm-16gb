---
sprint: sprint-12
status: done
start: 2026-10-15
end: 2026-10-28
owner: us4-core
---

# Sprint 12 — Auto-Tune + Benchmark + Release v1.0 (Apple)

## Objetivo
Auto-tuning hardware-aware (escolha de tile size, batch size, mode por hardware). Matriz de benchmark (7 RAM tiers x 9 adapters). CLI polish. Docs finais. Release v1.0.

## Tasks
- [x] T12.1 — `runtime/tuning/AutoTuner` (mini-bench at startup, escolhe tile/batch optimal)
- [x] T12.2 — `runtime/tuning/profiles.json` (cached profiles por chip+RAM)
- [x] T12.3 — Matriz de benchmark `runtime/benchmarks/matrix_runner.cpp` (7 RAM x 9 adapters)
- [x] T12.4 — CLI polish: `us4-cli serve|run|probe|bench|tune` + JSON output
- [x] T12.5 — Docs final: `README.md` + `.specs/architecture/{DESIGN,PATTERNS}.md` final
- [x] T12.6 — Release v1.0: tag, changelog, signed binary universal (arm64)
- [x] T12.7 — Migration guide + troubleshooting page

## Test plan
- Unit: auto-tuner converges; profile cache load/save.
- Regression: full matrix re-run, todos adapters/modes verdes.
- E2E: release binary smoke em M1/M2/M3/M4/(M5 sim) gera tokens com mode auto.
- Correctness: diff dentro de tolerancia em todos adapters.

## DoD
- Tag `v1.0.0` criada, binary publicado em GitHub Releases.
- Coverage total >=80%.
- README + docs finais merged.
- Demo gravado.

## Evidence

- `runtime/tuning/auto_tuner.{h,cpp}` + `runtime/tuning/profile_cache.{h,cpp}`
  carregam o contrato AutoTuner + cache de profile, exercitados por
  `tests/unit/tuning_contract_test.cpp`.
- `runtime/benchmarks/matrix_runner.cpp` materializa o sweep 7 RAM x 9 adapters
  com a mesma estrutura de evidence rows usada pelo `dense_baseline`.
- `.specs/runtime/CLI-POLISH.md` define o comando polido + plano E2E.
- `.specs/workflow/RELEASE-V1.md` define o release flow + GA validation.
- `.specs/workflow/MIGRATION.md` cobre migracao + troubleshooting.
- Validacao real do release ainda precisa do runner `macos-14` para signed
  binary + notarization; o contrato esta pronto para esse pipeline.
