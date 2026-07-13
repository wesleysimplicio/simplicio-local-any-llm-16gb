# Levantamento do projeto e viabilidade: LLM tier-1 em 16 GB de RAM

> Data: 2026-07-13. Branch: `claude/llm-project-assessment-tqlppc`.
> Escopo: inventário completo do repositório `ds4-simplicio-apple-v6` (código, specs, issues),
> análise honesta da pergunta "dá para rodar DeepSeek/GLM/Kimi em uma máquina de 16 GB
> com a tecnologia que levantamos?" e roadmap de próximos passos.

---

## 1. Sumário executivo

> **Revisão 2026-07-13 (mesma data, sessão posterior):** o item 1 abaixo assume pesos
> residentes em RAM. A auditoria do colibri (`docs/comparison-us4-vs-colibri.md`)
> provou viável o caminho alternativo — tronco denso residente + experts streamados do
> SSD — que torna checkpoints tier-1 completos executáveis em 16 GB **com lentidão
> extrema**. O plano de missão está em `docs/plan-simplicio-local-any-llm-16gb.md`.
> Os itens 2 e 3 permanecem válidos.

**Veredito curto:**

1. **Os checkpoints tier-1 completos (DeepSeek-V3/R1 671B, Kimi K2 ~1T, GLM-4.6 357B) não
   rodam em 16 GB de RAM** — nem com este runtime, nem com llama.cpp/MLX maduros. A conta
   não fecha por uma ordem de grandeza (seção 4). *(Ver nota de revisão acima: válido
   para pesos residentes em RAM; o streaming de experts muda a conclusão.)*
2. **As versões destiladas/compactas das mesmas famílias rodam bem em 16 GB**
   (DeepSeek-R1-Distill-Qwen-14B, GLM-4-9B, Kimi Moonlight-16B-A3B, etc. em Q4). Esse é o
   alvo realista e é exatamente para onde a arquitetura do projeto aponta.
3. **O runtime atual ainda não roda nenhum modelo real de nenhum tamanho.** A sessão da
   épica #81 tornou subsistemas isolados genuínos (loader safetensors, tokenizer BPE,
   forward denso, KV cache, MoE, quantização, speculative, serve nativo), mas tudo em
   escala de fixture (`hiddenSize = 8`, vocabulário de 4–16 tokens, modelos de ~426 bytes).
   O caminho entre o estado atual e "rodar um 7B–14B quantizado em 16 GB" está mapeado na
   seção 6.

---

## 2. Inventário: o que temos hoje

### 2.1 Estrutura do repositório

- **Runtime C++17/20** (`runtime/`): 19 módulos — core, cpu, neon, metal, mlx, ane,
  adapters, moe, kv, cache, memory, speculative, scheduler, net, telemetry, tuning,
  benchmarks.
- **CLI** (`apps/cli`): `us4-cli` com `--probe`, `run`, `serve` (proxy mlx_lm/Ollama) e
  `serve --native` (servidor OpenAI-compatible nativo).
- **Specs** (`.specs/`): visão de produto, DESIGN, PATTERNS, 9 ADRs (MLX primário,
  contrato de adapters, seleção de backend, BitNet packed, arquitetura KV, expert paging
  MoE, speculative decoding, ANE offload, load-balance de router), contratos de CLI,
  probe, telemetria e serve.
- **Sprints** (`.specs/sprints/`): 12 sprints planejados e executados, 75 task files.
- **Testes**: 222 unitários (GoogleTest/CTest) + 29 E2E (Playwright sobre o binário real).
  Validado nesta sessão em Linux x86_64: build limpo e 222/222 testes passando.
- **CI**: workflows desativados por decisão de custo (commit `b34a5a9`);
  DoD documentado em `.github/workflows/dod.yml` (desligado).

### 2.2 Estado real por subsistema (auditoria de código)

| Subsistema | Veredito | Evidência |
|---|---|---|
| Parser `.safetensors` | **Real, limitado** | Header binário + JSON + leitura de bytes, mas só dtype `F32`, sem mmap, sem shards, só `embedding.weight`/`lm_head.weight` (`safetensors_reader.cpp`, `model_asset.cpp:306`) |
| Parser `.gguf` | **Inexistente** | Enum reconhecido, binário nunca aberto (`model_asset.cpp:291-296`); fixtures são placeholders de 33 bytes — issue #102 |
| Tokenizer BPE | **Real, limitado** | Algoritmo de merges genuíno sobre `tokenizer.json` HF, mas pré-tokenização só whitespace: sem byte-level BPE, regex GPT-2, special tokens — insuficiente para DeepSeek/GLM/Kimi |
| Forward denso | **Real em escala toy** | Embedding + atenção single-head + projeção de saída com pesos reais quando disponíveis; `hiddenSize = 8` hardcoded (`dense_adapter_base.cpp:32`); **não há camadas de transformer (QKV/MLP/norm por camada)** |
| Matmul/atenção scalar e NEON | **Real** | Intrínsecos ARM genuínos com `vdotq_s32` (int8 dotprod); em x86 cai em fallback escalar explícito |
| Quantização int4/int8 | **Real, não ligada ao forward real** | Matemática groupwise correta e validada round-trip sobre pesos reais (#89), mas o forward com pesos reais pula a quantização (`dense_adapter_base.cpp:144-178`) |
| KV cache + SSD cold store | **Real em escala toy** | Tiering hot/warm/cold, spill/restore em disco genuíno, paridade de saída provada (#87); limite por contagem de páginas, não bytes |
| MoE expert paging | **Contábil** | `ExpertPager` é LFU de strings com limite em contagem (default 2), sem bytes, sem mmap (`expert_pager.cpp`); único peso de expert de fato aplicado: swap de `lm_head` no DeepSeek (#88) |
| Kimi/MiniMax/GLM adapters | **Sintéticos** | Mesma estrutura do DeepSeek, mas sem o swap de peso real — issue #103 |
| Metal | **Sintético** | "Planos" e contadores de dispatch; nenhum kernel GPU executa — issue #86 |
| MLX | **Sintético** | `EvaluateLastPlan()` só seta flag de sucesso (`mlx_bridge.cpp:41`) — issue #86 |
| ANE | **Sintético** | Contadores de compile/predict, scaffold declarado — issue #86 |
| Speculative decoding | **Real em escala toy** | Draft model real menor gera proposta autoregressiva genuína (#90) |
| Servidor OpenAI nativo | **Real, mínimo** | HTTP/1.1 POSIX single-thread, `/v1/models` + `/v1/chat/completions`, sem streaming SSE nem sampling params (#91) |
| Telemetria/tuning/scheduler | **Sintéticos** | Thermal `source="synthetic"`, batcher determinístico, caches contábeis |
| Benchmarks | **Placeholder** | `correctness_status=placeholder`; sem tokens/s nem memória de modelo real — issue #107 |

**Controle de memória existente:** nenhum budget em bytes. Os únicos limites são contagem
de experts (2) e contagem de páginas KV (4). Não há mmap, carregamento parcial de pesos
nem offload de pesos a disco.

### 2.3 Pontos fortes reais do projeto

- Disciplina de honestidade: fallbacks explícitos, flags `usedReal*`, documentação que não
  reivindica o que não existe (PROGRESS.md/GOAL_RESULT.md alinham com o código).
- Padrão de verificação forte: oráculo externo em Python independente (#82), testes de
  contrato por subsistema, E2E Playwright sobre o binário real.
- A arquitetura desenhada (ADRs de expert paging, KV tiering, BitNet/ternário, speculative,
  ANE opt-in) é exatamente a arquitetura correta para inferência em máquina de 16 GB — o
  que falta é a implementação em escala real, não o desenho.

---

## 3. Issues: estado consolidado

### 3.1 Abertas (9)

| Issue | Tema | Bloqueio |
|---|---|---|
| #81 | EPIC: runtime de inferência nativa 10/10 | Aberta até DoD completo |
| #86 | Backends Metal/MLX/ANE com execução real | **Bloqueada: precisa de hardware Apple Silicon/macOS** |
| #102 | GGUF real (parser binário + tensores) | Nenhum — portátil |
| #103 | Religar Kimi/MiniMax/GLM ao peso real de expert (só DeepSeek feito) | Nenhum |
| #104 | MoE real: FFN completa do expert (gate/up/down), não só lm_head | Nenhum |
| #105 | Caminho NEON/GQA do Llama ainda sintético (Q/K/V) | Parcial: validação plena exige ARM |
| #106 | Validar contra checkpoint de produção real (ex.: Qwen2.5-0.5B) | Precisa de download + oráculo Python |
| #107 | Benchmarks reais de performance/memória/estabilidade | Nenhum para CPU; Metal/MLX depende de #86 |
| #108 | CI com sanitizers (ASan/UBSan) + fuzzing dos parsers | Parcial: gate de CI depende de reativar workflows |

### 3.2 Fechadas (50)

- **Sprints 01–12** (#13–#24, #49–#76): fundações, CPU scalar, MLX/Metal skeleton, NEON
  hot paths, BitNet/ternário, arquitetura KV, Llama adapter, MoE foundation/advanced,
  batching + speculative, ANE offload, auto-tune + release v1.0 do scaffold.
- **Épica #81, filhas #82–#85 e #87–#91**: as 9 frentes "tornar real" descritas na seção 2.2,
  todas com PR squash-merged, testes novos e zero regressão (suíte cresceu de 203/16 para
  222/29 testes).
- #25: adoção do playbook de agente long-running.

---

## 4. A pergunta central: tier-1 em 16 GB é possível?

### 4.1 Os números dos modelos tier-1 (julho/2026)

| Modelo | Parâmetros (total/ativos) | Tamanho em disco | Menor quantização utilizável | Cabe em 16 GB? |
|---|---|---|---|---|
| DeepSeek-V3/R1 | 671B / 37B | ~720 GB (FP8) | ~131 GB (dynamic 1.58-bit Unsloth) | **Não** (8x acima) |
| Kimi K2/K2.5/K2.6 | ~1T / 32B | ~630 GB (FP8, QAT) | ~350 GB de RAM p/ Q2 | **Não** (20x acima) |
| GLM-4.6 | 357B / 32B | ~200 GB (Q4) | ~90–100 GB (Q2) | **Não** |
| GLM-4.5-Air | 106B / 12B | ~60 GB (Q4) | ~40 GB (Q2) | **Não** (2,5x acima) |

A matemática estrutural que mata a hipótese: mesmo com MoE + expert paging perfeito, os
**parâmetros ativos por token** desses modelos (32–37B) ocupam ~16–18 GB em Q4 — mais que a
RAM total da máquina antes de contar KV cache, ativações e sistema operacional. Como o
conjunto de experts ativos muda a cada token, um paging de disco perfeito ainda leria GBs
por token de NVMe, resultando em <0,1 token/s. Não é limitação do nosso runtime: é física
do modelo. Nenhum runtime existente (llama.cpp, MLX-LM, KTransformers) roda esses
checkpoints completos em 16 GB de forma utilizável.

### 4.2 O que efetivamente roda em 16 GB (o alvo certo)

Em uma máquina de 16 GB (ex.: Mac M1/M2/M3 base, memória unificada), o budget prático para
pesos + KV é ~10–12 GB. Cabem, das mesmas famílias tier-1:

| Modelo | Tipo | Tamanho Q4 aprox. | Observação |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-14B | denso destilado | ~9 GB | melhor raciocínio da família em 16 GB |
| DeepSeek-R1-0528-Qwen3-8B | denso destilado | ~5 GB | folga para contexto longo |
| GLM-4-9B / GLM-Z1-9B | denso | ~6 GB | família GLM em 16 GB |
| Kimi Moonlight-16B-A3B | **MoE** 16B/3B ativos | ~9–10 GB | MoE real que cabe — alvo ideal para o expert paging do projeto |
| Qwen2.5-0.5B | denso | ~0,4 GB | alvo de validação da issue #106 |

**Resposta à pergunta do usuário:** com a *arquitetura* que levantamos (quantização int4/int8,
expert paging, KV tiering com spill em SSD, speculative decoding, NEON/Metal/MLX/ANE), rodar
os **irmãos compactos/destilados** das famílias DeepSeek/GLM/Kimi em 16 GB é totalmente
viável — é o caso de uso para o qual o projeto foi desenhado. Rodar os **checkpoints
frontier completos** não é viável em 16 GB com nenhuma tecnologia conhecida hoje; o mínimo
prático para o DeepSeek-R1 completo quantizado é ~160–180 GB de RAM/VRAM.

---

## 5. Gap analysis: do estado atual até "rodar um 7B–14B em 16 GB"

O que falta, em ordem de dependência:

1. **Loader em escala real** — mmap (não `ifstream` para RAM), dtypes F16/BF16 (hoje só
   F32), índice de shards (`model.safetensors.index.json`), leitura de *todos* os tensores
   de camada (hoje só `embedding.weight`/`lm_head.weight`).
2. **Forward de transformer completo** — laço de camadas com RMSNorm, atenção multi-head
   real (QKV + RoPE + GQA), MLP (gate/up/down com SwiGLU), residuais; `hiddenSize`
   dinâmico do config (hoje constante 8). É o maior gap individual do projeto.
3. **Tokenizer completo** — byte-level BPE, regex de pré-tokenização, special tokens,
   chat template. Sem isso não há paridade token-a-token com nenhum modelo real.
4. **Quantização no caminho quente** — matmul int4/int8 direto sobre pesos quantizados
   (kernels NEON já existem; falta ligá-los ao forward real) e leitura de checkpoints já
   quantizados (GGUF #102).
5. **Budget de memória em bytes** — expert paging e KV paging por bytes com mmap, não por
   contagem; é o que transforma "cabe em 16 GB" em garantia de runtime.
6. **Metal/MLX reais (#86)** — única frente bloqueada por hardware; sem ela, o desempenho
   em Mac fica limitado a CPU/NEON.

---

## 6. Próximos passos recomendados (roadmap priorizado)

### Fase 1 — Escala real em CPU, portátil (desbloqueada agora, Linux ou Mac)

1. **Loader real em escala** (novo trabalho, prepara #106): mmap + F16/BF16 + shards +
   todos os tensores. Critério: abrir Qwen2.5-0.5B real sem estourar RAM.
2. **Forward de transformer completo** (novo trabalho, núcleo de #106): laço de camadas
   parametrizado pelo `config.json`. Critério: logits com diff dentro de tolerância vs
   `transformers` para N tokens.
3. **Tokenizer byte-level completo** (extensão de #84). Critério: paridade de ids com o
   tokenizer HF do modelo alvo.
4. **#106 — validação contra checkpoint real** (Qwen2.5-0.5B): fecha o ciclo 1–3 com
   oráculo externo. É o marco "o runtime roda um modelo de verdade".

### Fase 2 — Quantização e formatos (portátil)

5. **#102 — GGUF real**: abre acesso direto ao ecossistema de checkpoints quantizados
   (é onde estão os Q4 prontos dos modelos da seção 4.2).
6. **Quantização no forward** (fecha o gap deixado por #89): matmul int8/int4 sobre pesos
   reais. Critério: 7B–9B Q4 gerando texto coerente em <12 GB de RSS.
7. **#108 — sanitizers/fuzzing locais** nos parsers (safetensors/gguf/json/tokenizer),
   antes de aceitar arquivos de modelo arbitrários da internet.

### Fase 3 — MoE real em bytes (portátil; alvo: Moonlight-16B-A3B)

8. **#104 — FFN completa do expert** e **#103 — religar Kimi/GLM/MiniMax**.
9. **Expert paging por bytes** (evolução do `ExpertPager`): mmap dos shards de expert +
   budget em bytes configurável (`--memory-budget`). Critério: MoE 16B/A3B rodando em
   <12 GB com eviction observável.

### Fase 4 — Apple Silicon (bloqueada por hardware: exige Mac/runner macos-14)

10. **#86 — Metal/MLX/ANE reais** + **#105 — caminho GQA/NEON validado em ARM**.
11. **#107 — benchmarks reais** (tokens/s, memória, estabilidade) com metodologia
    documentada, comparando scalar vs NEON vs Metal/MLX.
12. Reativar CI macOS (decisão de custo do usuário) e fechar o DoD da **épica #81**.

### Decisões que precisam do dono do projeto

- **Hardware Apple Silicon**: sem um Mac (ou runner `macos-14` pago), as fases 4 e o DoD
  da épica ficam permanentemente bloqueados. Alternativa: manter alvo CPU/NEON e validar
  em Linux ARM (Graviton/Raspberry Pi 5) como proxy parcial.
- **Reativação do CI**: #108 e o gate de DoD dependem disso (custo).
- **Modelo alvo oficial de 16 GB**: recomendação — Qwen2.5-0.5B para validação (#106),
  DeepSeek-R1-Distill-Qwen-14B Q4 como alvo denso final, Kimi Moonlight-16B-A3B como alvo
  MoE final.

---

## 7. Fontes

- Auditoria de código desta sessão (caminhos e linhas citados na seção 2.2).
- Issues #81–#108 do repositório.
- Tamanhos/requisitos dos modelos: Unsloth (DeepSeek-R1 dynamic 1.58-bit, ~131 GB;
  DeepSeek-V3.1 local), gulla.net (R1 671B local com 32 GB+swap, inutilizável na prática),
  apxml.com e lushbinary.com (Kimi K2.5/K2.6: FP8 ~630 GB, Q2 ~350 GB RAM),
  artificialanalysis.ai (GLM-4.6 vs Kimi K2). Links no PR/relatório da sessão.
