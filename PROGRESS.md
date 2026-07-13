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

## Checkpoint seguinte (3)

Status: done

Task:
#85 - Forward denso real (embedding/projeção a partir de pesos carregados,
sem DeterministicValue)

Result:
`BuildTokenEmbedding`/`BuildOutputProjection` em `DenseAdapterBase` agora
consultam `asset->realTensors` (populado por #83) e usam o lookup real de
`embedding.weight`/`lm_head.weight` quando o shape bate exatamente com
`hiddenSize`/`vocabulary.size()`, com fallback explícito e per-tensor (não
mais um flag global `usingRealWeights`) para o caminho sintético
determinístico anterior. Isso cobre qwen/gemma/bitnet/ternary/llama
(caminho não-NEON)/deepseek/kimi/minimax/glm, já que todos passam pelo
`DenseAdapterBase::Generate` compartilhado. O CLI expõe
`used_real_weights` (texto/JSON).

Duas rodadas de revisão adversarial (cpp-reviewer + security-reviewer via
agents) encontraram e corrigiram, antes do merge:
- shape check da projeção de saída estava transposto (não batia com a
  convenção real HF/safetensors `[vocab_size, hidden_size]` para
  `nn.Linear.weight`) — corrigido com transposição explícita em
  `TryRealOutputProjection`;
- a flag de "usando pesos reais" era global por asset em vez de por
  chamada, o que suprimia ruído sintético mesmo quando o dado daquela
  chamada específica ainda era sintético — corrigido com `usedReal`
  per-call;
- quantização int8/int4 podia ser aplicada por engano a pesos reais se o
  manifesto irmão declarasse esse dtype — corrigido pulando quantização
  quando a fonte é real;
- recursão sem limite no parser JSON (`json_value.cpp`) permitia DoS por
  stack overflow via `tokenizer.json`/header safetensors malicioso —
  corrigido com limite de profundidade;
- underflow de `size_t` em `data_offsets` malformado
  (`safetensors_reader.cpp`) podia causar alocação absurda/crash —
  corrigido com validação `end >= begin` e verificação contra o tamanho
  real do arquivo;
- cast de `double` fora de faixa/infinito para `size_t` (UB) — corrigido
  com validação `std::isfinite`+faixa antes do cast.

Evidência ponta-a-ponta: fixture `tests/fixtures/models/toy-dense-real/`
(safetensors real + manifesto) faz o CLI gerar exatamente o token previsto
por um oráculo externo (embedding one-hot, lm_head com margem clara),
coberto por um novo teste Playwright.

Validation:
`cmake --build build`; `ctest --test-dir build --output-on-failure` (212/212,
zero regressão nos testes pré-existentes); `npx playwright test --reporter=list`
(26/26); `clang-format --dry-run --Werror` e `clang-tidy -p build` limpos nos
arquivos tocados.

Next:
#81.7 (MoE real) e #81.9 (speculative real) podem reusar `realTensors`/
`hasRealWeights` agora que o forward denso base os consome de fato. #81.6
(KV cache) pode ganhar teste de paridade com/sem cache sobre este forward
real.

## Checkpoint seguinte (4)

Status: done

Task:
#87 - KV cache real sobre estados de atenção reais

Result:
`kv_cache_real_forward_contract_test.cpp` chama `Generate()` duas vezes com
o mesmo `RuntimeContext` e o mesmo prompt sobre a fixture `toy-dense-real`
(pesos reais): a primeira chamada tem `kvCacheHit=false` (popula o pager),
a segunda tem `kvCacheHit=true` (reusa), e a saída (`generatedTokens`/
`text`) é idêntica nas duas — provando que o cache não altera o resultado
sobre um forward real, não apenas sobre o caminho sintético anterior.

Validation:
`cmake --build build`; `ctest --test-dir build --output-on-failure` (213/213);
`npx playwright test --reporter=list` (26/26); `clang-format --dry-run --Werror`
e `clang-tidy -p build` limpos no arquivo novo.

Next:
#81.7 (MoE real) e #81.9 (speculative real) são os próximos passos.

## Checkpoint seguinte (5)

Status: done

Task:
#89 - Quantização aplicada a pesos reais (não à projeção sintética)

Result:
`QuantizeProjectionInt8`/`QuantizeProjectionInt4` (e `BuildGroupScales`)
foram extraídas do namespace anônimo de `dense_adapter_base.cpp` para
`runtime/cpu/quantize_projection.{h,cpp}`, reutilizáveis por testes fora do
adapter. `quantization_real_weights_contract_test.cpp` carrega
`embedding.weight`/`lm_head.weight` reais (fixture `toy_real.safetensors`,
#81.2) e roda o round-trip quantize→dequantize real (`DequantizeInt8Groups`/
`DequantizeInt4Groups`), validando que o erro por elemento fica dentro de
uma tolerância documentada (até um `scale` completo, cobrindo o erro de
arredondamento de meio-passo mais folga). Isso é distinto do pipeline de
geração, que (desde #81.4/#81.5) pula quantização quando a fonte já é
real, para não degradar pesos exatos.

Validation:
`cmake --build build`; `ctest --test-dir build --output-on-failure` (215/215,
zero regressão); `npx playwright test --reporter=list` (26/26);
`clang-format --dry-run --Werror` e `clang-tidy -p build` limpos nos
arquivos tocados.

Next:
#81.7 (MoE real) e #81.9 (speculative real) permanecem como próximos
passos; #81.10 (API nativa) depende deles.

## Checkpoint seguinte (6)

Status: done

Task:
#90 - Speculative decoding real — modelo draft real, não cópia mockada

Result:
`TryRealDraftProposal` (novo helper em `dense_adapter_base.cpp`) carrega o
`draft_model_path` do asset via `LoadModelAsset`/`SafetensorsReader` e, se
o modelo draft tiver `embedding.weight`/`lm_head.weight` reais com shape
compatível com o vocabulário compartilhado, roda um forward autoregressivo
genuíno (embedding lookup + dot-product argmax, sem attention — um draft
model é feito pra ser barato) para produzir a proposta especulativa,
posição a posição a partir do token anterior do PRÓPRIO modelo draft. Isso
substitui o mock anterior (`draftProposal = authoritativeTokens` com o
último token incrementado). Fallback explícito e visível
(`used_real_draft_model` no CLI) para o comportamento sintético anterior
quando o draft model não tem pesos reais compatíveis — sem mudança de
comportamento pra nenhuma fixture existente.

Fixture `toy-dense-real-draft.safetensors` (hidden_size=2, menor que o
modelo alvo, como um draft model real seria) foi desenhada para que seu
forward real, com o token anterior "delta", também argmaxe para "delta" —
coincidindo com a saída real do modelo alvo (calculada offline,
independentemente). O teste de contrato usa isso pra provar que
`speculativeAcceptedTokens`/`speculativeAcceptanceRate` refletem uma
aceitação baseada em computação real, não um mock roteirizado garantido
pra sempre aceitar ou rejeitar.

Validation:
`cmake --build build`; `ctest --test-dir build --output-on-failure` (216/216,
zero regressão); `npx playwright test --reporter=list` (26/26, incluindo o
teste de #85 que reusa o mesmo manifesto agora com draft model real);
`clang-format --dry-run --Werror` e `clang-tidy -p build` limpos nos
arquivos tocados.

Next:
#81.7 (MoE real) é a última frente tratável sem hardware Apple. #81.10 (API
nativa) depende dela + de #81.4/#81.5 (já concluídas).

## Checkpoint seguinte (7)

Status: done

Task:
#88 - MoE real — streaming de pesos de expert do disco, roteamento
aplicado ao forward

Result:
`TryLoadExpertShardLmHead` (novo em `runtime/core/model_asset.{h,cpp}`) lê
de verdade o tensor `lm_head.weight` do shard `.safetensors` real apontado
por `asset.expertShardPaths[expertIndex]` (via `SafetensorsReader`, não
mais um manifesto texto nunca aberto). `DeepSeekMoEAdapter::Generate` usa
isso: quando o expert escolhido pelo router tem shard real compatível, o
adapter substitui o `lm_head.weight` do asset roteado pelo peso real
daquele expert antes de chamar `DenseAdapterBase::Generate` — o resultado
da geração passa a refletir de fato o peso do expert selecionado, não só
uma contabilidade de `Touch()` no pager. CLI expõe
`used_real_expert_weights`.

Fixture `toy-moe-real/` tem um tensor "decoy" no modelo base (argmaxa para
"alpha") e um shard de expert real com peso diferente (argmaxa para
"beta"). O teste de contrato prova que a saída observada é "beta" — ou
seja, o peso do expert roteado, não o decoy da base, dirigiu o forward.

Apenas o DeepSeek foi religado nesta sessão; kimi/minimax/glm compartilham
o mesmo padrão e podem adotar o mesmo helper depois.

Validation:
`cmake --build build`; `ctest --test-dir build --output-on-failure` (217/217,
zero regressão); `npx playwright test --reporter=list` (26/26);
`clang-format --dry-run --Werror` e `clang-tidy -p build` limpos nos
arquivos tocados.

Next:
Religar kimi/minimax/glm ao mesmo helper de expert real. #81.10 (API
nativa) já tem todas as dependências centrais (#81.2/#81.4/#81.5)
concluídas.

## Checkpoint seguinte (8)

Status: done

Task:
#91 - API OpenAI-compatible nativa (sem depender de mlx_lm.server/Ollama)
+ benchmarks reais

Result:
`us4-cli serve --native` sobe um servidor HTTP/1.1 real e mínimo
(`runtime/net/native_http_server.{h,cpp}`, sockets POSIX puros, sem
dependência nova) que responde `GET /v1/models` e
`POST /v1/chat/completions` diretamente do `Generate()` do runtime, via
`runtime/net/openai_chat_handler.{h,cpp}` — sem processo externo, sem
Python. O handler aceita um campo de extensão `model_path` (fora do schema
OpenAI, documentado como extensão us4-cli) para carregar pesos reais como
`us4-cli run --model-path` já fazia, e a resposta expõe
`used_real_weights`.

Evidência ponta-a-ponta real (não mock): teste Playwright novo
(`tests/e2e/us4-cli-serve.spec.ts`) sobe o binário `us4-cli serve --native`
como processo real, faz requisição HTTP genuína com `model_path` apontando
pra fixture `toy-dense-real` (#85), e recebe `used_real_weights: true` com
`content: "delta"` — a mesma previsão do oráculo externo já usada como
evidência em #85, agora servida por um servidor HTTP nativo de verdade.
Testes unitários cobrem o parser de request e o handler isoladamente (sem
sockets).

README (`6.2.1`) distingue explicitamente `serve` (proxy pra
mlx_lm.server/Ollama via `scripts/openai_serve.py`) de `serve --native`
(motor nativo direto). Benchmark de performance Metal/MLX real fica
bloqueado pelo mesmo hardware Apple Silicon que falta pra #81.5 — o que
foi verificado aqui é corretude funcional, documentado como tal, não
throughput.

Validation:
`cmake --build build` (sem warning novo); `ctest --test-dir build
--output-on-failure` (222/222, zero regressão); `npx playwright test
--reporter=list` (29/29, incluindo 3 testes novos do servidor nativo);
`clang-format --dry-run --Werror` e `clang-tidy -p build` limpos nos
arquivos tocados.

Next:
Das 10 frentes da epic #81, 8/9 issues filhas tratáveis estão fechadas.
Só #81.5 (Metal/MLX/ANE reais) permanece aberta, bloqueada por hardware
(requer macOS/Apple Silicon real — não pode ser implementada nem validada
nesta sessão Linux x86_64).

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

### Checkpoint 13

Status: done

Task:
#103 [#81.7b] - Rewire kimi/minimax/glm MoE adapters to apply the routed
expert's real lm_head.weight, the same way #81.7/#88 already wired
DeepSeekMoEAdapter.

Result:
`KimiMoEAdapter`, `MiniMaxMoEAdapter` e `GlmMoEAdapter` agora chamam
`TryLoadExpertShardLmHead` e substituem `lm_head.weight` pelo shard real do
expert selecionado pelo router antes de delegar para
`DenseAdapterBase::Generate`, igual ao padrao ja usado pelo DeepSeek em
#88. Cada familia recebeu fixtures proprias (`toy-moe-real-kimi`,
`toy-moe-real-minimax`, `toy-moe-real-glm`) com decoy na base e um expert
shard cujo `lm_head.weight` argmaxa para um token diferente -- provando que
o peso real do expert, e nao o decoy da base, dirigiu a saida observada.
`embedding.weight` das fixtures usa a mesma linha one-hot em todas as
posicoes do vocabulario (nao so na linha "alpha"), porque o texto de
route-signature de cada familia ("kimi-route e1", etc.) tokeniza para
palavras fora do vocabulario e cai no hash `TokenIdFor`, cujo indice
resultante variava por familia -- sem essa normalizacao o teste ficava
acoplado a um detalhe de hash em vez de provar a substituicao do peso do
expert. `moe_real_expert_weights_contract_test.cpp` ganhou 3 testes novos
(Kimi/MiniMax/Glm), cada um usando `FindAdapterByModel` para a familia
correspondente. Zero regressao: suite unitaria foi de 222 para 225 testes
verdes.

Validation:
`cmake --build build`; `clang-format --dry-run --Werror` nos arquivos
tocados; `clang-tidy -p build` nos 3 adapters; `ctest --test-dir build
--output-on-failure` (225/225 verde).

Next:
Avaliar #102 [#81.2b] (GGUF real parsing) ou #104 [#81.7c] (MoE roteando
pela FFN completa do expert, nao so pela output projection).

### Checkpoint 14

Status: done

Task:
#102 [#81.2b] - GGUF real parsing (parser binario real do container GGUF,
nao so tratar a extensao como dica e nunca abrir o arquivo).

Result:
Novo `runtime/core/gguf_reader.{h,cpp}` (`GgufReader`) parseia o container
GGUF binario de verdade: magic `GGUF`, versao, contagem de tensors/kv,
secao de metadata (com skip generico por tipo, incluindo arrays de um
nivel, e decodificacao real de `general.alignment` quando presente), array
de tensor info (nome, dims, ggml_type, offset), e secao de dados alinhada
(`align_up` pelo alignment lido ou 32 por padrao). `ReadFloat32` le bytes
reais de tensors ggml F32. Segue a mesma postura defensiva do
`SafetensorsReader` (#81.5/#85): caps em tamanho de string, contagem de
array, numero de dimensoes e contagem de tensors/kv contra arquivo
malformado/adversario, e validacao de offset+tamanho contra o tamanho real
do arquivo antes de ler.

`LoadModelAsset` agora tambem tenta `GgufReader::Open` para arquivos
`.gguf` (mesmo padrao ja usado para `.safetensors` desde #83), populando
`realTensors`/`realTensorShapes`/`hasRealWeights` quando `embedding.weight`
ou `lm_head.weight` sao encontrados como F32. Os fixtures `.gguf`
pre-existentes (`toy-qwen.gguf`, `toy-llama.gguf`, `toy-bitnet.gguf`) sao
placeholders de texto, nao GGUF real -- continuam falhando o parse (como
esperado) e caindo no fallback synthetic/hidratacao por manifest sibling,
sem regressao.

Fixture novo: `tests/fixtures/gguf/toy_real.gguf` (gerado por
`generate_toy_gguf.py`, mesmos valores/shapes do `toy_real.safetensors` de
#83 para comparação direta entre os dois formatos). Testes novos:
`gguf_reader_contract_test.cpp` (4 testes: rejeita placeholder, parseia
header/shapes reais, le bytes float32 reais, erro explicito pra tensor
ausente) e `ModelAssetContractTest.RealGgufAssetExposesRealTensorBytes`
em `model_asset_contract_test.cpp`. Zero regressao: suite unitaria foi de
225 para 230 testes verdes.

Validation:
`cmake --build build`; `clang-format --dry-run --Werror` nos arquivos
tocados; `clang-tidy -p build` em `gguf_reader.cpp`/`model_asset.cpp`
(sem warnings); `ctest --test-dir build --output-on-failure` (230/230
verde).

Next:
Avaliar #104 [#81.7c] (MoE roteando pela FFN completa do expert) ou #105
[#81.4b] (path NEON/GQA do Llama ainda sintetico).

### Checkpoint 15

Status: done

Task:
#105 [#81.4b] - o path NEON/GQA-especifico do Llama
(`BuildQueryRow`/`BuildKeyRow`/`BuildValueRow` + a projecao de saida dentro
de `LlamaAdapter::Generate`) chamava `BuildTokenEmbedding`/
`BuildOutputProjection` SEM passar o `asset` -- entao `usedRealWeights`
ficava sempre falso e pesos reais nunca eram usados nesse path, mesmo
quando a fixture tinha `embedding.weight`/`lm_head.weight` reais.

Result:
`BuildQueryRow`/`BuildKeyRow`/`BuildValueRow` agora recebem `asset` e um
`usedReal` out-param, seguindo o mesmo contrato do
`DenseAdapterBase::BuildTokenEmbedding`. `BuildValueRow` só aplica a
perturbacao sintetica posicional quando o embedding NAO e real (mesma
logica de gate per-call ja usada em `DenseAdapterBase::Generate`).
`LlamaAdapter::Generate` agora passa `request.asset` em toda chamada de
Build*/`BuildOutputProjection`, agrega um `anyRealWeightUsed` e passa isso
pro `FinalizeGenerationResult` (que já aceitava esse parametro, só nao
era usado pelo Llama). `MaterializeProjectionForAsset` (metodo publico de
`DenseAdapterBase`) ganhou um `sourceIsRealWeights` opcional pra evitar
quantizar/dequantizar pesos reais s? porque o manifest declara int8/int4
pra outro caminho.

Durante a implementacao, uma revisao adversarial contra o proprio oraculo
externo (embedding one-hot -> RoPE identity na posicao 0 -> attention
trivial de 1 kv-row -> lm_head coluna 0) pegou um bug real introduzido na
mesma edicao: o replace acidentalmente removeu as duas linhas que copiavam
`keyBuffer`/`valueBuffer` pros tensors `key`/`value` antes do
`GqaAttention`, deixando esses tensors com lixo/zero. O teste previu
"beta" e observou "delta" -- a divergencia expôs a omissao antes do merge,
nao depois.

Fixture novo: `tests/fixtures/models/toy-llama-real/` (hidden_size=4,
query_heads=1, kv_heads=1, head_dim=4 -- kv width igual ao hidden size de
proposito, pra que a MESMA `embedding.weight` sirva de fonte real pra
query, key e value). Teste novo:
`AdapterGenerationContractTest.LlamaNeonPathUsesRealEmbeddingAndLmHeadWeights`
em `adapter_generation_contract_test.cpp`, com oraculo externo (nao
reusa nenhum codigo do runtime pra prever "beta"). Zero regressao: suite
unitaria foi de 230 para 231 testes verdes.

Validation:
`cmake --build build`; `clang-format --dry-run --Werror` nos arquivos
tocados; `clang-tidy -p build` em `llama_adapter.cpp`/
`dense_adapter_base.cpp` (sem warnings); `ctest --test-dir build
--output-on-failure` (231/231 verde).

Next:
Avaliar #104 [#81.7c] (MoE roteando pela FFN completa do expert), #106
[#81.11] (validar contra checkpoint real), #107 [#81.12] (benchmarks
reais), ou #108 [#81.13] (sanitizers/fuzzing locais).

### Checkpoint 16

Status: done

Task:
#108 [#81.13] - build com ASan/UBSan disponivel localmente + harness de
fuzzing pros parsers de entrada nao confiavel (`JsonValue::Parse`,
`SafetensorsReader`, `GgufReader`, `BpeTokenizer::LoadFromFile`). Job de CI
fica documentado mas nao ativado (decisao explicita do usuario de manter
`.github/workflows.disabled` desligado nesta sessao).

Result:
`CMakeLists.txt` ganhou `US4_ENABLE_ASAN`/`US4_ENABLE_UBSAN` (mesmo padrao
do `US4_ENABLE_COVERAGE` que ja existia) e `US4_BUILD_FUZZERS`. Novo
`runtime/fuzz/` com 4 harnesses libFuzzer (`us4_fuzz_json_value`,
`us4_fuzz_safetensors_reader`, `us4_fuzz_gguf_reader`,
`us4_fuzz_bpe_tokenizer`) -- os tres parsers path-based escrevem o input
num arquivo temporario por chamada (`ScopedFuzzInputFile`) porque a API
deles le de disco, nao de buffer.

Rodar os harnesses de verdade (nao so compilar) contra os fixtures
existentes como seed corpus PEGOU UM BUG REAL em minutos:
`GgufReader::ReadFloat32` multiplicava as dimensoes do shape sem checar
overflow/limite antes de `std::vector<float> values(elementCount)` --
um shape adversario faz esse construtor lancar `std::length_error`
(exception nao capturada, abort do processo). Corrigido com checagem de
overflow na multiplicacao e um limite explicito contra
`vector::max_size()` antes de alocar, retornando o erro explicito de
sempre em vez de crashar. Dois testes de regressao novos em
`gguf_reader_contract_test.cpp` fixam esse caso (overflow puro e
"cabe em size_t mas excede max_size()") pra nao depender só do fuzzer
pra pegar de novo.

Validado localmente com Clang 18 + `libclang-rt-18-dev` (pacote extra
necessario nesta sandbox pra runtime do libFuzzer/ASan/UBSan): os 4
harnesses rodaram ~25s cada contra os fixtures existentes como seed
corpus, sem nenhum outro crash apos a correcao. Suite unitaria normal
(sem sanitizers): 231 para 233 testes verdes (os 2 testes de regressao),
zero regressao.

Validation:
```
cmake -S . -B build-fuzz -G Ninja -DUS4_BUILD_FUZZERS=ON \
  -DUS4_ENABLE_ASAN=ON -DUS4_ENABLE_UBSAN=ON -DUS4_BUILD_TESTS=OFF \
  -DUS4_BUILD_BENCHMARKS=OFF -DCMAKE_CXX_COMPILER=clang++
cmake --build build-fuzz
./build-fuzz/runtime/fuzz/us4_fuzz_<nome> -max_total_time=25 <seed-corpus-dir>
```
`cmake --build build` (build normal); `clang-format --dry-run --Werror` e
`clang-tidy -p build` nos arquivos tocados (sem warnings); `ctest
--test-dir build --output-on-failure` (233/233 verde).

Next:
Avaliar #104 [#81.7c] (MoE roteando pela FFN completa do expert), #106
[#81.11] (validar contra checkpoint real), ou #107 [#81.12] (benchmarks
reais).

### Checkpoint 17

Status: done

Task:
#107 [#81.12] - nenhuma das issues #82-#91 media performance (throughput,
latencia, memoria), so corretude funcional. DoD da epic #81 pede
benchmarks reais e comparacao explicita real vs sintetico.

Result:
Novo `runtime/benchmarks/real_forward_throughput.cpp`: mede tokens/s,
latencia de decode (mean/min/max sobre 5 repeticoes) e RSS do processo
sobre o forward REAL (fixtures `toy-dense-real`/`toy-llama-real`, pesos
reais de #85/#105) lado a lado com o MESMO adapter no caminho totalmente
sintetico (`asset=nullptr`), pros casos qwen/scalar, qwen/neon-requested e
llama/neon-gqa. Metodologia documentada em
`runtime/benchmarks/README.md` (hardware, modelo, quantizacao -- nenhuma,
fp32 pra isolar o custo do tensor real do custo de dequant -- contexto,
decodificacao determinística sem temperatura).

Leitura honesta registrada: o caminho real fica ~8-38% mais lento que o
sintetico nesses fixtures pequenos (custo de indexar tensor real vs gerar
valor deterministico inline) -- esperado, nao regressao. RSS
praticamente nao muda entre antes/depois das 5 repeticoes. Documentado
explicitamente que os numeros absolutos NAO extrapolam pra um modelo de
producao (isso seguindo aberto em #81.11/#106) e que NEON so acelera de
fato em ARM64 real (esta sandbox e x86_64 sem NEON, entao os casos
`neon-requested` mostram fallback correto, nao aceleracao).

Nao mudou nenhum codigo de runtime, so um benchmark novo + doc -- sem
impacto na suite unitaria (233/233 continua verde).

Validation:
`cmake --build build`; `clang-format --dry-run --Werror` e `clang-tidy -p
build` em `real_forward_throughput.cpp` (sem warnings); `ctest
--test-dir build --output-on-failure` (233/233 verde, sem novos testes
pois este e um benchmark, nao um contract test); execucao real do
benchmark com output capturado no README.

Next:
Avaliar #104 [#81.7c] (MoE roteando pela FFN completa do expert) ou #106
[#81.11] (validar contra checkpoint real -- feasibility incerta:
requer download de rede e/ou compute significativo).
