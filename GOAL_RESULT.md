# Goal Result

## Summary (sessao atual)

Auditoria completa da issue #81 (EPIC "runtime de inferencia nativa 10/10"):
nenhuma das 10 frentes declaradas e real de ponta a ponta hoje. Decompus a
epic em 10 issues filhas honestas (#82-#91), cada uma com veredito
REAL/PARCIAL/SINTETICO e evidencia de codigo. Fechei a #84 (tokenizer BPE
real) nesta sessao: adicionei um parser JSON minimo, um tokenizer BPE real
que consome tokenizer.json de verdade, uma fixture de BPE treinado
genuinamente com oraculo independente em Python, e wiring explicito no
adapter que reporta quando cai para o tokenizer ingenuo (nunca disfarçado).

As demais 9 frentes seguem abertas. #85 (Metal/MLX/ANE reais) esta
bloqueada por ambiente: esta sessao roda em Linux x86_64 sem Metal/MLX, e
por isso nao pode implementar nem validar essa frente sem um runner
macOS/Apple Silicon real — reportar sucesso ali seria fabricar evidencia.

## Previous Summary (T11.5/T11.6)

T11.5 foi concluida. A evidencia de benchmark ANE foi ampliada no
`dense_baseline`: os casos `ane-requested` para Qwen e Llama ficam visiveis e
cada caso passa a registrar mixed-dispatch, contadores ANE e estado termico. Em
host nao-M5, a saida esperada e fallback observavel.

T11.6 foi concluida. Foi adicionada cobertura Playwright para garantir que
`--backend ane` em host nao elegivel exponha fallback explicito e mantenha o
mixed dispatch desabilitado.

## Previous Summary

T11.3 e T11.4 foram concluidas. O runtime agora tem um coordenador explicito
de mixed dispatch entre Metal e ANE, alem de um `ThermalMonitor` que aplica
downgrade de modo sob pressao termica derivada do probe. O estado nativo e o
CLI projetam estrategia mixed-dispatch, atividade ANE e sinais termicos sem
esconder fallback.

## Changed Files

- runtime/ane/mixed_dispatch.h
- runtime/ane/mixed_dispatch.cpp
- runtime/adapters/dense_adapter_base.cpp
- runtime/core/runtime_context.h
- runtime/core/runtime_context.cpp
- runtime/core/ius4v6_adapter.h
- apps/cli/main.cpp
- tests/unit/runtime_acceleration_contract_test.cpp
- tests/unit/runtime_contract_runner.cpp
- runtime/tuning/thermal_monitor.h
- runtime/tuning/thermal_monitor.cpp
- runtime/tuning/README.md

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
- Sprint 12 segue como proximo bloco de issues abertas.
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
