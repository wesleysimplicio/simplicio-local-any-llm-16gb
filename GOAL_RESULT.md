# Goal Result

## Summary

T11.3 e T11.4 foram concluidas. O runtime agora tem um coordenador explicito
de mixed dispatch entre Metal e ANE, alem de um `ThermalMonitor` que aplica
downgrade de modo sob pressao termica derivada do probe. O estado nativo e o
CLI projetam estrategia mixed-dispatch, atividade ANE e sinais termicos sem
esconder fallback.

## Changed Files

- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\runtime\ane\mixed_dispatch.h
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\runtime\ane\mixed_dispatch.cpp
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\runtime\adapters\dense_adapter_base.cpp
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\runtime\core\runtime_context.h
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\runtime\core\runtime_context.cpp
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\runtime\core\ius4v6_adapter.h
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\apps\cli\main.cpp
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\tests\unit\runtime_acceleration_contract_test.cpp
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\tests\unit\runtime_contract_runner.cpp
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\runtime\tuning\thermal_monitor.h
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\runtime\tuning\thermal_monitor.cpp
- C:\Users\wesley.simplicio\Pictures\m\us4-v6-simplicio-apple\runtime\tuning\README.md

## Validation Commands

```bash
npm run lint
npm test -- --coverage
npm run pack:dry
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
npx playwright test --reporter=list,html tests/e2e/us4-cli.spec.ts
build\runtime\benchmarks\dense_baseline.exe
```

## Validation Results

- build: pass
- tests: pass
- lint: pass
- e2e: pass

## Remaining Risks

- T11.5-T11.6 ainda seguem pendentes no planejamento local.
- A fonte termica atual e `probe-derived`; leitura real de IOPMrootDomain/powermetrics fica como aprofundamento Apple-host.

## Suggested PR Title

`feat(ane): add mixed dispatch and thermal monitor`

## Suggested PR Body

```md
## Summary
- add mixed Metal/ANE dispatch coordinator and runtime integration
- add probe-derived thermal monitor with explicit downgrade semantics
- expose mixed dispatch and thermal telemetry in native generation results and CLI
- cover ANE-eligible, metal-fallback, and thermal downgrade plans in native contract tests

## Validation
- [x] lint
- [x] unit and regression
- [x] native build and ctest
- [x] Playwright CLI E2E
- [x] pack dry-run

## Risks
- ANE benchmark evidence and graceful fallback slices still remain for Sprint 11
```
