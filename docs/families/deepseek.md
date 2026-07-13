# Familia DeepSeek-V3 / R1 no motor `engine/c/glm.c`

> Issue [#120](https://github.com/wesleysimplicio/ds4-simplicio-apple-v6/issues/120), epica
> [#116](https://github.com/wesleysimplicio/ds4-simplicio-apple-v6/issues/116) (rodar LLM
> tier-1 em 16GB de RAM sem GPU). Depende de #117 (vendoring do motor colibri, mergeado) e
> #119 (perfil 16GB, mergeado). Bloqueada para rodar em checkpoint real / hardware real de
> 16GB por [#118](https://github.com/wesleysimplicio/ds4-simplicio-apple-v6/issues/118) — ver
> "Limitacoes" abaixo.

## Por que DeepSeek-V3/R1 "quase" ja funcionava

O motor `glm.c` foi escrito para GLM-5.2 (`glm_moe_dsa`), mas GLM-5.2 **herda a arquitetura
MLA + MoE routed/shared do DeepSeek-V3** quase tensor-a-tensor: mesmos nomes de peso
(`self_attn.q_a_proj`, `self_attn.kv_a_proj_with_mqa`, `mlp.gate.weight`,
`mlp.gate.e_score_correction_bias`, `mlp.shared_experts.*`, `mlp.experts.{i}.{gate,up,down}_proj.weight`),
mesma semantica de router sigmoid + bias de correcao ("noaux_tc"). A diferenca que faltava:
GLM-5.2 usa **n_group=1** (todos os experts num unico "grupo", i.e. nenhum group-limiting);
DeepSeek-V3/R1 real usa **n_group=8, topk_group=4** (256 experts, 8 grupos de 32, so' 4 grupos
concorrem pelos top-8). Antes desta issue, `load_cfg()` abortava incondicionalmente se
`n_group != 1` — o motor rejeitava qualquer config DeepSeek-V3/R1 real na porta de entrada.

DeepSeek-V3/R1 tambem **nao tem** o indexer DSA (exclusivo do `glm_moe_dsa` de GLM-5.2) — o
motor ja detecta isso automaticamente (`has_dsa=0` quando os tensores do indexer nao existem
no snapshot) e cai no caminho de atencao MLA densa, o mesmo caminho ja validado para GLM-5.2
com seq curta. A cabeca MTP (multi-token prediction) segue o mesmo padrao de deteccao
automatica (`has_mtp`) e e' opcional/independente desta issue.

## Semantica exata do router (confirmada lendo `transformers`)

Fonte: `transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3TopkRouter.forward`
(mesma classe usada por `DeepseekV3ForCausalLM`, instanciada e inspecionada neste ambiente).

```python
scores = router_logits.sigmoid()                              # [tokens, E]
scores_for_choice = scores + e_score_correction_bias            # bias so' aqui
group_scores = scores_for_choice.view(-1, n_group, E // n_group).topk(2, dim=-1)[0].sum(-1)
group_idx = topk(group_scores, k=topk_group)                    # grupos vencedores
score_mask = expand(group_idx) -> [tokens, E] booleano
scores_for_choice = scores_for_choice.masked_fill(~score_mask, -inf)
topk_indices = topk(scores_for_choice, k=top_k)                 # SELECAO usa bias
topk_weights = scores.gather(topk_indices)                      # PESO usa sigmoid SEM bias
if norm_topk_prob: topk_weights /= (topk_weights.sum(-1) + 1e-20)
topk_weights *= routed_scaling_factor
```

Pontos que uma implementacao ingenua erra facilmente:

1. **Score de grupo = soma dos TOP-2 (nao top-1, nao media, nao soma total) scores COM bias**
   de cada expert do grupo.
2. **Selecao de grupo e' `topk` puro** (sem sorted, ties resolvidos por ordem de indice —
   irrelevante pra corretude numerica).
3. **Mascara os experts fora dos grupos vencedores para `-inf`** — nao zera, nao penaliza:
   remove da corrida do top-k final por completo.
4. **A selecao final do top-k usa `scores_for_choice` (COM bias)**, mas **o peso do gate
   usa `scores` (sigmoid puro, SEM bias)** — o bias e' um truque de *load balancing* que so'
   influencia QUAL expert e' escolhido, nunca QUANTO peso ele recebe. Aplicar o bias no peso
   tambem e' o erro mais comum e o mais dificil de notar (os tokens ainda "fazem sentido",
   so' ficam sutilmente errados).
5. `n_group == 1` e' o caso de GLM-5.2: um unico grupo == todos os experts == a mascara
   nunca exclui nada == comportamento legado, bit-identico.

## Matriz de diferencas de config: DeepSeek-V3/R1 vs GLM-5.2 (`glm_moe_dsa`)

| campo (`config.json`)     | GLM-5.2 (`glm_moe_dsa`)        | DeepSeek-V3/R1 real           | Tiny fixture desta issue (`deepseek_tiny/`) |
|----------------------------|--------------------------------|--------------------------------|----------------------------------------------|
| `n_group`                  | `1`                             | `8`                             | `4`                                            |
| `topk_group`               | `1`                             | `4`                             | `2`                                            |
| `n_routed_experts`         | grande (centenas)               | `256`                           | `8`                                            |
| `num_experts_per_tok`      | poucos                          | `8`                             | `2`                                            |
| indexer DSA (`index_topk`, `index_n_heads`, `index_head_dim`) | presente (lightning indexer) | **ausente** (nao existe nesta familia) | ausente (`has_dsa=0` auto-detectado) |
| MTP (`model.layers.<n_layers>.*`) | opcional (--mtp na conversao) | opcional (multi-token prediction real, mesmo padrao de nomes) | ausente neste fixture (`has_mtp=0` auto-detectado) |
| Nomenclatura de tensor MLA/MoE | `self_attn.q_a_proj`, `mlp.experts.{i}.*`, `mlp.gate.e_score_correction_bias` | **identica** | identica |
| Peso do gate usa bias?     | nao (so' sigmoid; mascara de grupo e' no-op com n_group=1) | nao (so' sigmoid; bias so' filtra a selecao) | nao (mesma regra, exercitada de fato) |

A ultima linha e' a mesma regra nos dois casos — a diferenca real e' inteiramente o
group-limiting (linhas 1-4 da tabela), que so' tem efeito observavel quando `n_group>1`.

## O que foi implementado em `engine/c/glm.c`

- `load_cfg()`: removido o `exit(1)` incondicional em `n_group!=1`. `n_group`/`topk_group`
  ausentes no `config.json` (GLM-5.2 nunca os seta) caem no default legado `1`/`1` via
  `gi()` retornando `0` -> normalizado para `1`. Validacao `CKR` adicionada:
  `n_group` em `[1, n_experts]`, `topk_group` em `[1, n_group]`, e um check explicito de
  `n_routed_experts % n_group == 0` (grupos de tamanho igual, como o `.view()` do PyTorch
  exige).
- `moe()`: antes do loop de selecao top-k de cada posicao, quando `n_group>1`:
  1. calcula `gscore[g]` = soma dos top-2 `choice[e]` (sigmoid+bias) de cada grupo `g`;
  2. seleciona os `topk_group` grupos de maior `gscore` (mesmo padrao de selecao linear
     "exclui os ja escolhidos" ja usado no resto do arquivo, sem alocacao extra por token);
  3. mascara `choice[e] = -infinito` para todo `e` fora dos grupos selecionados.
  O restante do `moe()` (selecao top-k, peso do gate = `logit[best]` sigmoid puro sem bias,
  `norm_topk_prob`, `routed_scaling_factor`, batch-union de experts, shared expert) **nao foi
  tocado** — e' exatamente por isso que `n_group==1` fica bit-identico ao comportamento
  anterior (o bloco de masking inteiro nem executa).
- Buffers extras (`gscore`, `gsel`) sao alocados **so'** quando `n_group>1` (`NULL` e custo
  zero em GLM-5.2).

## Validacao (oraculo tiny, sem checkpoint real)

Sem acesso a um checkpoint DeepSeek-R1/V3 real (centenas de GB) nem a uma maquina fisica de
16GB neste ambiente, a validacao usa o mesmo padrao ja estabelecido para GLM-5.2
(`engine/c/tools/make_glm_oracle.py` / `ref_glm.json`): um modelo **real**
`transformers.DeepseekV3ForCausalLM` (arquitetura real, nao uma reimplementacao), pesos
aleatorios, dimensoes minusculas, mas com **`n_group=4`, `topk_group=2`, 8 experts** — ou
seja, o group-limiting real esta' ativo e testado, nao contornado.

- Gerador: `engine/c/tools/make_deepseek_oracle.py` (mesmo padrao de
  `make_glm_oracle.py`) -> `engine/c/deepseek_tiny/` (pesos + `config.json`) +
  `engine/c/ref_deepseek.json` (prompt_ids, full_ids greedy, tf_pred teacher-forcing).
- Teste automatizado: `engine/c/tests/test_deepseek_family.py` (roda em `make test`,
  padrao `test_16gb_profile.py`) — teacher-forcing 32/32 e geracao greedy 20/20 do binario
  `./glm` (precisao f32/f16, sem quantizacao) contra o oraculo `transformers`.
- **Prova de que o group-limiting e' de fato exercitado** (nao um no-op por coincidencia):
  o mesmo fixture com `n_group=1`/`topk_group=1` forcado no `config.json` (ignorando o
  masking de grupo real) da' **22/32** no teacher-forcing — ou seja, sem a logica de
  group-limiting implementada nesta issue, o motor erraria 10 das 32 posicoes neste
  fixture. Com o group-limiting correto: 32/32.
- Regressao GLM-5.2 (`n_group=1`): `ref_glm.json` continua 32/32 teacher-forcing e 20/20
  geracao greedy; suite completa `tests/test_16gb_profile.py` (8/8) continua verde.
- `engine/c/tools/convert_fp8_to_int4.py --indir deepseek_tiny --outdir <out> --ebits 4
  --io-bits 8 --n-layers 4`: gera um container int4 valido que o motor carrega e executa
  sem crash (`classify()`/`convert_shard()` sao genericos por nome de tensor — a familia
  DeepSeek usa os MESMOS nomes de GLM-5.2 para MLA/MoE, entao nenhuma mudanca de logica foi
  necessaria; so' o `--n-layers` precisa refletir a contagem real de camadas da familia:
  78 para GLM-5.2, **61 para DeepSeek-V3/R1 real**, 4 para este fixture tiny). Em precisao
  int4 sobre pesos aleatorios pequenos o teacher-forcing diverge do oraculo f32 (mesmo
  efeito, na mesma magnitude, ja' observado em `glm_tiny` a int4 — ruido de quantizacao
  sobre `normal(0, 0.05)`, esperado e sem relacao com o group-limiting); o criterio de
  aceite desta issue e' o container carregar/rodar sem abortar, nao bater bit-a-bit em
  int4 num fixture aleatorio.

## Limitacoes reais deste ambiente

- Nenhum checkpoint DeepSeek-V3/R1 real (~650GB/~715GB em FP8) foi baixado nem usado — esta
  issue nao fabrica esse resultado. A validacao e' inteiramente contra o oraculo tiny
  `transformers`, que prova a **corretude da matematica do router e do forward MLA/MoE**,
  nao o desempenho/qualidade em escala real.
- Rodar um checkpoint DeepSeek-V3/R1 real dentro de um teto de RSS de 16GB (perfil
  `docs/profiles/16gb.md`, gate `COLI_RSS_CEILING_GB`/`COLI_PROFILE=16gb`) depende de
  hardware real de 16GB para validar em condicoes reais — bloqueado por
  [#118](https://github.com/wesleysimplicio/ds4-simplicio-apple-v6/issues/118), fora do
  escopo cumprivel nesta issue/ambiente. O gate de RSS e' agnostico de arquitetura (projeta
  a partir do modelo efetivamente carregado, testado em #119) e portanto ja' cobre
  DeepSeek-V3/R1 sem mudanca — falta apenas a maquina real para medir o numero final.
