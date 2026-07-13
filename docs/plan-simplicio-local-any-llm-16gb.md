# Plano de missão: simplicio-local-any-llm-16gb

> Objetivo declarado pelo dono do projeto: **rodar qualquer LLM tier-1 (DeepSeek, GLM,
> Kimi) localmente em uma máquina de até 16 GB de RAM — nem que seja com lentidão
> extrema.**
>
> Data: 2026-07-13. Contexto: `docs/assessment-llm-tier1-16gb.md` (estado do US4) e
> `docs/comparison-us4-vs-colibri.md` (auditoria do colibri).

---

## 1. Por que agora é "sim" (revisão do veredito anterior)

O assessment anterior respondeu "não cabe em 16 GB" assumindo pesos residentes em RAM.
O colibri (Apache-2.0, fork em `wesleysimplicio/colibri`) prova o caminho que muda a
resposta: **não é preciso caber — é preciso residir só o tronco denso e streamar os
experts do disco.**

Física do MoE que viabiliza:

- Num MoE de fronteira, só o **tronco denso** (atenção, shared experts, embeddings,
  ~11–17B parâmetros) é usado em todo token. Em int4 isso dá **~6–10 GB residentes**.
- Os **experts roteados** (95%+ dos parâmetros) mudam a cada token e podem viver no
  SSD, lidos sob demanda (~19 MB por expert no container int4 do colibri), com cache
  LRU + hot-set fixado + page cache do SO.
- O KV cache MLA comprimido custa ~2 KB/token — irrelevante no budget.

Evidência medida (colibri, GLM-5.2 744B int4): 9.9 GB residentes, RSS 14.1 GB numa
máquina de 24 GB a 0.07–0.11 tok/s; 1.06 tok/s num M5 Max. **Em 16 GB ninguém
demonstrou ainda — essa é exatamente a nossa missão e o nosso diferencial.**

O custo é velocidade: com pouca RAM para cache de experts, o decode é disk-bound.
Ordem de grandeza esperada em 16 GB: **0.03–0.1 tok/s** em NVMe comum (3 GB/s),
**0.1–0.4 tok/s** em SSD de Mac (6–14 GB/s), melhorando com hot-set aprendido e
`top-p` agressivo. Minutos por resposta. O dono aceitou explicitamente essa troca.

## 2. Budget de memória alvo (máquina de 16 GB)

| Componente | Budget |
|---|---|
| SO + desktop + margem anti-swap | ~3 GB |
| Tronco denso residente (int4; int3/int2 seletivo se preciso) | ≤ 8.5 GB |
| Cache de experts (LRU + pin) | 1.5–3 GB |
| KV (MLA comprimido) + buffers/slabs + tokenizer | ≤ 1 GB |
| **Teto de RSS do processo** | **≤ 13 GB** |

Regra dura: nunca tocar swap (escrita desgasta SSD e derruba o sistema); o auto-budget
por `MemAvailable` do colibri já existe e vira gate.

## 3. Viabilidade por modelo alvo

| Modelo | Total/ativos | Tronco denso (int4, est.) | Disco (int4) | Em 16 GB |
|---|---|---|---|---|
| GLM-5.2 | 744B / ~40B | ~9.9 GB (medido) → ~7.5 GB com int3 seletivo | ~370 GB | Viável (apertado; caso âncora) |
| DeepSeek-V3/R1 | 671B / 37B | ~8–9 GB (estimar em M2) | ~340 GB | Viável — **exige group-limited routing (`n_group=8`) que o colibri não tem** |
| Kimi K2 | ~1T / 32B | ~6–7 GB (estimar em M2) | ~550 GB | Viável em RAM; **disco é o gargalo** |
| Qwen3-235B-A22B (bônus) | 235B / 22B | ~4–5 GB | ~120 GB | Confortável; bom segundo alvo |
| Densos gigantes (ex. 405B) | tudo ativo | n/a | ~200 GB | Só modo demo: streaming de camadas, minutos/token |

Pré-requisito de hardware que precisa ficar explícito na doc de usuário: além dos
16 GB de RAM, são necessários **400–600 GB livres de SSD rápido** (interno ou
USB4/Thunderbolt) e tolerância a térmica de leitura sustentada.

## 4. Estratégia de código — decisão principal (pendente do dono)

Duas rotas para o motor:

- **Rota A (recomendada): adotar o motor do colibri como base do engine tier.**
  Apache-2.0 permite (com atribuição/NOTICE). O forward MoE, kernels quantizados,
  streaming, KV MLA e MTP já estão escritos e validados token-exact contra oráculo.
  Nosso trabalho vira: perfil 16 GB, novas famílias, Metal, provas/CI e produto —
  em cima de uma base que já entrega o resultado.
- **Rota B: reimplementar em C++ no runtime US4**, usando o colibri como
  referência/oráculo. Mantém o código 100% nosso e o padrão C++/CMake do repo, mas
  refaz ~3.7k linhas de C de alta densidade técnica antes de gerar o primeiro token.

Recomendação técnica: **Rota A** para chegar ao resultado, mantendo o US4 como camada
de produto/aceleração (CLI, probe, telemetria, testes, Metal) — e migração incremental
posterior se fizer sentido. Registrar a escolha como **ADR-010** (regra do repo:
decisão irreversível vira ADR).

## 5. Marcos (cada um com critério mensurável)

### M0 — Validar a premissa no hardware real (custo ~zero de código)
Rodar o colibri como está numa máquina de 16 GB (ou VM/cgroup com RAM capada em 16 GB)
com o GLM-5.2 int4 pré-convertido do Hugging Face.
**Critério:** `coli doctor` verde ou diagnóstico do bloqueio; se rodar: RSS, tok/s e
hit-rate registrados = baseline oficial a bater.
**Risco conhecido:** o auto-cap pode reduzir o cache a ponto de `doctor` reprovar;
nesse caso M1 vira obrigatório, não otimização.

### M1 — Perfil 16 GB dedicado ("16gb-profile")
Auto-budget agressivo: teto de RSS 13 GB como gate duro; tronco denso com quantização
mista (int4 → int3/int2 nas matrizes menos sensíveis — kernels int2 já existem);
MTP desligado por padrão (a frio custa mais experts/token); DSA ligado; pin aprendido
persistente; `--temp 0.7 --topp 0.7` default do perfil.
**Critério:** numa máquina 16 GB, GLM-5.2 responde um prompt de chat com ≥ 20 tokens
corretos, RSS ≤ 13 GB, zero swap, sem OOM-kill, em 3 execuções consecutivas.

### M2 — "Any tier-1": DeepSeek e Kimi
Generalizar o motor: group-limited routing (`n_group>1`, `topk_group`,
`e_score_correction_bias`) para DeepSeek-V3/R1; dims/config do Kimi K2 (mesma família
MLA); remover tetos de buffer fixos (`sc[8192]`, `row[8192]`) com validação de config;
conversor FP8→int4 parametrizado por família.
**Critério:** oráculo tiny por família (DeepSeek, Kimi, GLM) com teacher-forcing
token-exact 32/32 + geração 20/20; e pelo menos um checkpoint real de cada família
gerando texto coerente na máquina de 16 GB.

### M3 — Provas e CI (superar a maior fraqueza do colibri)
GitHub Actions (quando o dono religar o CI, decisão de custo): build C + suíte
completa (incluindo forward e tokenizer, hoje fora do `make test` do colibri),
oráculos tiny das 3 famílias, sanitizers ASan/UBSan + fuzzing dos parsers
(padrão da issue #108), harness de benchmark reproduzível e medição de qualidade da
quantização (perplexity int4 vs BF16 — o colibri admite nunca ter medido).
**Critério:** badge verde; qualquer regressão de correção quebra o build.

### M4 — Apple Silicon: Metal no tronco denso (issue #86; exige Mac real)
O caminho warm é matmul-bound (datapoint 9950X: 57% matmul). Portar o matmul
quantizado do tronco denso + indexer DSA para Metal (simdgroup int8/int4), buffers
compartilhados via unified memory (zero-copy), NEON-SDOT permanece para o resto.
**Critério:** no mesmo Mac, tok/s do nosso build > colibri CPU-only; marca de
referência pública a bater: 1.06 tok/s (M5 Max, 128 GB). Meta secundária: melhor
tok/s já registrado em um Mac de 16 GB.

### M5 — "Any LLM" literal: fallback para densos gigantes
Streaming de camadas por mmap para modelos densos que não cabem (working set = 1
camada + ativações). Modo demo explícito (`--mode demo-dense`), com expectativa
honesta de minutos/token.
**Critério:** um denso ≥ 70B gera 10 tokens corretos em 16 GB sem OOM.

### M6 — Produto simplicio
Integração no `us4-cli` (probe/doctor/plan unificados), serve OpenAI com SSE atrás do
binário nativo, Web UI, E2E Playwright cobrindo o fluxo chat/serve (nosso padrão),
README com números reproduzíveis e a distinção implemented/experimental/fallback
exigida pela épica #81.
**Critério:** DoD da épica #81 fechável, com evidência.

## 6. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Disco: 370–550 GB por modelo numa máquina pequena | Documentar pré-requisito; suportar SSD externo USB4/TB; um modelo por vez |
| Térmica/desgaste de SSD em leitura sustentada | Monitorar temperatura no doctor; leitura é read-only (não desgasta como escrita); alertar QLC |
| Swap-storm em 16 GB | Gate duro de RSS ≤ 13 GB; abortar com mensagem clara em vez de degradar |
| Qualidade int3/int2 no tronco denso | Medir perplexity por configuração (M3) antes de virar default |
| Pesos GLM-5.2/K2: downloads de centenas de GB | Conversor resumable shard-a-shard (já existe); mirrors HF pré-convertidos |
| `n_group>1` e novas famílias introduzem bugs sutis | Oráculo tiny por família em CI antes de tocar checkpoint real (M2→M3) |
| Buffers de pilha fixos herdados | Eliminar em M2 com validação de config estendida |
| Hardware Apple para M4 | Sem Mac/runner macos-14, M4 fica bloqueado (mesmo status da issue #86) |

## 7. Decisões pendentes do dono

1. **Rota A (adotar motor colibri, Apache-2.0) vs Rota B (reimplementar em C++)** —
   recomendação: A, registrada em ADR-010.
2. **Máquina de validação de 16 GB** (qual hardware é o alvo canônico: Mac 16 GB,
   notebook x86 16 GB, ou ambos) e **disco disponível** (400–600 GB).
3. **Religar CI** (custo) — pré-requisito do M3.
4. **Acesso a Mac/Apple Silicon** — pré-requisito do M4/#86.
5. Onde o trabalho vive: neste repo (runtime Apple-first) ou num repo novo
   `simplicio-local-any-llm-16gb` com este repo como camada de produto Apple.
