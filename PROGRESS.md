# Progress Log

## Current Status

Auditoria da issue #81 (EPIC) concluida: nenhuma das 10 frentes do runtime e
real de ponta a ponta (achados detalhados nas issues filhas #82-#91). Issue
#84 (tokenizer BPE real) fechada nesta sessao como primeira frente tratavel
em ambiente Linux sem hardware Apple. Demais frentes seguem abertas com
bloqueio de ambiente explicito quando dependem de Metal/MLX/ANE reais
(macOS/Apple Silicon).

## Latest Checkpoint

Status: done

Task:
#84 - Tokenizer real (BPE) fiel ao modelo, consumindo tokenizer.json

Result:
`BpeTokenizer` (runtime/core/bpe_tokenizer.{h,cpp}) parseia tokenizer.json
real (schema HuggingFace `tokenizers`, model.type=BPE + vocab + merges) via
um parser JSON minimo proprio (`runtime/core/json_value.{h,cpp}`) e executa
o algoritmo BPE genuino (merge por menor rank, nao whitespace/lowercase
fake). `DenseAdapterBase::TokenizePrompt` usa o tokenizer real quando o
asset tem um tokenizer.json com vocab/merges reais, e cai explicitamente
para o tokenizer ingenuo quando nao tem — o CLI agora expoe
`used_real_bpe_tokenizer` e `tokenizer_fallback_reason` (texto/JSON), nunca
disfarcando o fallback. Fixture genuina de BPE treinado
(`tests/fixtures/tokenizer/toy_bpe_tokenizer.json`, gerado por
`generate_toy_bpe.py`) mais um oraculo independente em Python
(`reference_output.json`) validam paridade token-a-token no
`bpe_tokenizer_contract_test.cpp`.

Validation:
`cmake --build build`; `ctest --test-dir build --output-on-failure` (206/206);
`npx playwright test --reporter=list` (25/25); `clang-format --dry-run --Werror`
nos arquivos tocados; `clang-tidy -p build` nos arquivos tocados.

Next:
Issue #82 (oraculo real) e #83 (loader real de pesos) sao os proximos passos
tratáveis sem hardware Apple. #85 (Metal/MLX reais) permanece bloqueada por
ambiente (requer macOS/Apple Silicon real).

## Checkpoint seguinte

Status: done

Task:
#83 - Loader real de pesos (.safetensors) — parsear e carregar tensores de
verdade

Result:
`SafetensorsReader` (runtime/core/safetensors_reader.{h,cpp}) parseia o
header binario real do formato safetensors (8 bytes de tamanho + JSON de
tensores) e le os bytes reais de tensores F32 do corpo do arquivo, em vez de
tratar a extensao como dica sem nunca abrir o binario. `ModelAsset` ganhou
`realTensors`/`hasRealWeights` e `LoadModelAsset` tenta carregar
`embedding.weight`/`lm_head.weight` reais quando o arquivo `.safetensors` e
genuino, registrando `safetensors_load_status` explicito quando cai para
placeholder. Fixture genuina binaria (`tests/fixtures/safetensors/
toy_real.safetensors`, gerada por `generate_toy_safetensors.py`) com dois
tensores reais valida shapes/dtype/bytes em
`safetensors_reader_contract_test.cpp`.

Validation:
`cmake --build build`; `ctest --test-dir build --output-on-failure` (210/210);
`npx playwright test --reporter=list` (25/25); `clang-format --dry-run --Werror`
e `clang-tidy -p build` limpos nos arquivos tocados.

Next:
GGUF real (subconjunto do formato) e wiring pleno do forward denso (#81.4)
para consumir `realTensors` em vez de `DeterministicValue`.

## Checkpoint seguinte (2)

Status: done

Task:
#82 - Contrato de correção e oráculo real (não escalar-vs-escalar)

Result:
`oracle_correctness_contract_test.cpp` carrega os MESMOS pesos reais do
fixture `toy_real.safetensors` (via `SafetensorsReader`) e roda o kernel
nativo (`ScalarMatmul` sobre embedding lookup real) contra
`oracle_reference_logits.json` — logits calculados por uma implementação
Python independente (`generate_oracle_reference.py`, parser+matmul escritos
do zero, sem reusar nenhum código C++ do runtime). Isso fecha o gap
apontado na auditoria: antes o "oráculo" comparava NEON com o próprio
kernel escalar do runtime; agora compara com uma referência externa sobre
pesos reais. Teste adicional garante falha explícita quando os pesos não
carregam (sem fallback mascarado para escalar-vs-escalar).

Validation:
`cmake --build build`; `ctest --test-dir build --output-on-failure` (212/212);
`clang-format --dry-run --Werror` e `clang-tidy -p build` limpos no arquivo
novo.

Next:
#81.4 (forward denso real na pipeline de produção dos adapters) é o proximo
passo natural para levar esse mesmo padrao de oraculo para a geração
ponta-a-ponta, não apenas para um matmul isolado.

Status: done

Task:
T11.4 - Build thermal monitor surface

Result:
`ThermalMonitor` entrou em `runtime/tuning` com amostragem derivada do probe,
decisao de downgrade e exposicao no `RuntimeContext`. O runtime agora limita
modos sob pressao `elevated` e `critical`, e o `us4-cli run` mostra
`thermal_pressure_level`, `thermal_reason` e `thermal_downgraded`.

Validation:
`npm run lint`; `npm test -- --coverage`; `npm run pack:dry`;
`cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html tests/e2e/us4-cli.spec.ts`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
T11.5 - adicionar evidencia de benchmark ANE/Metal e documentar limites em
hosts sem M5 real.

Status: done

Task:
T11.3 - Build mixed metal ane dispatch surface

Result:
`MixedDispatchCoordinator` entrou em `runtime/ane`, o `RuntimeContext` passou
a expor esse coordenador, e o caminho ANE do `DenseAdapterBase` agora executa
um plano explicito de estagios Metal/ANE em vez de cair num scaffold opaco. O
`GenerationResult` e o `us4-cli run` agora mostram estrategia, contagem de
estagios Metal/ANE, layers compiladas e chamadas de predicao para manter o
fallback visivel.

Validation:
`npm run lint`; `npm test -- --coverage`; `npm run pack:dry`;
`cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html tests/e2e/us4-cli.spec.ts`

Next:
T11.4 - adicionar `ThermalMonitor` e gates de degradacao termica.

Status: done

Task:
T11.2 - Build ane layer offloader surface

Result:
`LayerOffloader` entrou com heuristica explicita para attention/MLP estaticos,
mantendo embedding, router, low-bit e shapes nao-estaticos fora do caminho ANE.
O `RuntimeContext` agora expõe essa decisao e o dense path ANE consegue
materializar compile intent so para camadas elegiveis.

Validation:
`npm run lint`; `npm test -- --coverage`; `npm run pack:dry`;
`cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`

Next:
T11.3 - coordenar mixed dispatch entre Metal e ANE.

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

### Checkpoint 11

Status: in_progress

Task:
T11.5 - Add ane benchmark evidence

Result:
`dense_baseline` agora inclui casos `ane-requested` para Qwen e Llama fixture,
alem de emitir estrategia mixed-dispatch, contadores ANE e sinais termicos por
caso. Em hosts sem M5+, a evidencia valida e fallback explicito, nao numero
sintetico.

Validation:
`clang-format --dry-run --Werror runtime\\benchmarks\\dense_baseline.cpp`;
`clang-tidy -p build runtime\\benchmarks\\dense_baseline.cpp`;
`cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npm run lint`; `npm test -- --coverage`; `npm run pack:dry`;
`npx playwright test --reporter=list,html tests/e2e/us4-cli.spec.ts`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar issue #68 e seguir para T11.6.

### Checkpoint 12

Status: in_progress

Task:
T11.6 - Harden ane graceful fallback

Result:
Sprint 11 foi sincronizada para marcar T11.6 como concluida e o Playwright
ganhou cobertura explicita para `--backend ane` em host nao elegivel. O teste
exige `requested-backend-unavailable`, `fallback=true` e
`mixed_dispatch_strategy=disabled` quando ANE nao esta disponivel.

Validation:
`npm run lint`; `npm test -- --coverage`; `npm run pack:dry`;
`cmake --build build --config Release`;
`ctest --test-dir build --output-on-failure -C Release`;
`npx playwright test --reporter=list,html tests/e2e/us4-cli.spec.ts`;
`build\\runtime\\benchmarks\\dense_baseline.exe`

Next:
Fechar issue #69 e seguir para Sprint 12.

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
