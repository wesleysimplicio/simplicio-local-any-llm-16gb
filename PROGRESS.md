# Progress Log

## Current Status

Sprint 10 em execucao. O fechamento de speculative decoding agora entrou na
telemetria nativa/CLI, com acceptance rate, accepted/rejected tokens e fallback
deterministico visiveis para os caminhos com draft model.

## Latest Checkpoint

Status: done

Task:
T10.6 - Expand speculative decoding telemetry

Result:
A telemetria speculative agora projeta `speculative_strategy`,
`speculative_session_scope`, acceptance rate, accepted/rejected tokens e
fallback token no resultado nativo e no CLI para os fluxos com draft model.

Validation:
`npm run lint`; `npm test -- --coverage`; `npm run pack:dry`;
`cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html tests/e2e/us4-cli.spec.ts`

Next:
Sprint 11 - abrir `T11.1` para o backend ANE M5+.

## Checkpoints

### Checkpoint 1

Status: done

Task:
T08.3 - Deepen deepseek moe adapter

Result:
DeepSeek agora deriva logits do prompt/manifesto, materializa a assinatura
`moe-route eX eY`, e expoe reuse/eviction do `ExpertPager` de forma visivel no
output nativo e no CLI.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
T08.4 - replicar o mesmo nivel de concretude no Kimi e seguir para T08.5.

### Checkpoint 2

Status: in_progress

Task:
T08.4 - Deepen kimi moe adapter

Result:
Kimi ja consome `RouterDecision`, injeta `kimi-route eX eY`, mostra reuse do
pager no mesmo `RuntimeContext`, e ganhou cobertura unit + E2E dedicada.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T08.4 e seguir para T08.5 (loader MoE shard-aware).

### Checkpoint 3

Status: in_progress

Task:
T08.5 - Add moe shard-aware loader contract

Result:
`ModelAsset` agora preserva shard metadata (`moe_lazy_load`,
`moe_active_experts`, `expertShardPaths`) e o CLI nativo projeta isso como
`moe_shard_count`, `moe_active_experts` e `moe_lazy_load`.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T08.5 e seguir para T08.6 (telemetria MoE expandida).

### Checkpoint 4

Status: in_progress

Task:
T08.6 - Expand moe telemetry surface

Result:
O CLI nativo e o benchmark baseline agora expoem `moe_hit_rate`,
`moe_eviction_rate` e `moe_router_entropy`, enquanto `TelemetrySnapshot`
ganhou helpers semanticos para presenca/hit/eviction de MoE.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T08.6 e encerrar a issue #20 antes de seguir para Sprint 09.

### Checkpoint 5

Status: in_progress

Task:
T09.1 - Add minimax adapter surface

Result:
`MiniMaxMoEAdapter` entrou no runtime nativo, registrado no adapter registry,
com assinatura `minimax-route eX eY`, fixture `minimax-m2` shard-aware e
contratos unit/native/E2E para manter o adapter visivel.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T09.1 e seguir para T09.2 (GLM adapter surface).

### Checkpoint 6

Status: in_progress

Task:
T09.2 - Add glm adapter surface

Result:
`GlmMoEAdapter` entrou com fixture shard-aware, assinatura `glm-route eX eY`,
telemetria MoE no CLI e coverage nativa/E2E para manter a familia visivel.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T09.2 e seguir para T09.3 (speculative prefetch).

### Checkpoint 7

Status: in_progress

Task:
T09.3 - Build speculative expert prefetch

Result:
`SpeculativePrefetch` entrou como contrato family-scoped com `hitRatio`,
`hitCount`, `missCount` e protecao explicita contra wrong-expert leakage.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T09.3 e seguir para T09.4 (sparsity-aware cache).

### Checkpoint 8

Status: in_progress

Task:
T09.4 - Build sparsity-aware cache surface

Result:
`SparsityAwareCache` entrou em `runtime/cache`, foi ligado ao
`RuntimeContext`, e os adapters MoE agora projetam hit/miss, `patternHash` e
`patternKey` no resultado nativo e no CLI.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T09.4 e seguir para T09.5 (multimodal cache).

### Checkpoint 9

Status: in_progress

Task:
T09.5 - Build multimodal cache surface

Result:
`MultimodalCache` entrou em `runtime/cache`, foi ligado apenas ao
`MiniMaxMoEAdapter`, e o CLI nativo agora expõe hit/miss/modalities sem vazar
essa semântica para os caminhos dense-only.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T09.5 e seguir para T09.6 (telemetria consolidada de prefetch
e sparsity).

### Checkpoint 10

Status: in_progress

Task:
T09.6 - Expand advanced moe telemetry

Result:
Os adapters MoE agora projetam `moe_prefetch_*` junto de
`moe_sparsity_cache_hit_rate`, e o CLI nativo/bench passaram a expor essa
telemetria para inspeção direta.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T09.6 e encerrar a issue #21 antes de seguir para Sprint 10.

### Checkpoint 8

Status: in_progress

Task:
T09.4 - Build sparsity-aware cache surface

Result:
`SparsityAwareCache` entrou em `runtime/cache`, foi ligado ao
`RuntimeContext`, e os adapters MoE agora projetam hit/miss, `patternHash` e
`patternKey` no resultado nativo e no CLI.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T09.4 e seguir para T09.5 (multimodal cache).

### Checkpoint 9

Status: in_progress

Task:
T09.5 - Build multimodal cache surface

Result:
`MultimodalCache` entrou em `runtime/cache`, foi ligado apenas ao
`MiniMaxMoEAdapter`, e o CLI nativo agora expõe hit/miss/modalities sem vazar
essa semântica para os caminhos dense-only.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T09.5 e seguir para T09.6 (telemetria consolidada de prefetch
e sparsity).

Result:
`GlmMoEAdapter` entrou no runtime nativo, registrado no adapter registry,
com assinatura `glm-route eX eY`, fixture `glm-5.1` shard-aware, inferencia de
familia no loader e contratos unit/native/E2E para manter o adapter visivel.

Validation:
`npm run lint`; `npm test -- --coverage`; `cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html`; `npm run pack:dry`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar PR/merge de T09.2 e seguir para T09.3 (speculative prefetch).

### Checkpoint 7

Status: in_progress

Task:
T09.3 - Build speculative expert prefetch

Result:
`SpeculativePrefetch` entrou em `runtime/moe` com plano de prefetch por familia,
reconciliacao com a rota real, `hitRatio` estavel e garantia contratual de
`wrongExpertLeakPrevented`.

Validation:
`clang-format --dry-run --Werror`; `clang-tidy -p build runtime/moe/speculative_prefetch.cpp`;
`cmake --build build --config Release`; `ctest --test-dir build --output-on-failure -C Release`;
`npm run lint`; `npm test -- --coverage`; `npm run pack:dry`

Next:
Fechar PR/merge de T09.3 e seguir para T09.4 (sparsity-aware cache).

## Blockers

Nenhum bloqueio funcional. O ambiente local continua sem GTest instalado, entao
os gates nativos seguem por `us4_runtime_smoke_test` e
`us4_runtime_contract_runner`.

## Validation History

| Command | Result | Notes |
|---|---|---|
| `npm run lint` | pass | JS starter ok |
| `npm test -- --coverage` | pass | 13 testes verdes |
| `cmake --build build --config Release` | pass | rebuild nativo ok |
| `ctest --test-dir build --output-on-failure -C Release` | pass | smoke + contract runner |
| `npx playwright test --reporter=list,html` | pass | 16 testes verdes na rodada atual |
| `npx playwright test --reporter=list,html` | pass | 17 testes verdes com loader MoE shard-aware |
| `npx playwright test --reporter=list,html` | pass | 17 testes verdes com telemetria MoE expandida |
| `npm run pack:dry` | pass | tarball `0.1.27` ok |
| `build\\runtime\\benchmarks\\dense_baseline.exe` | pass | observabilidade MoE/low-bit ok |
