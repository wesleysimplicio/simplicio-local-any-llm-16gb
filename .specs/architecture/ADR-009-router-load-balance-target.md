# ADR-009: `Corrigir alvo de balanceamento do RouterLoadBalanceLoss para 1/N`

---

## Status

`Aceito`

---

## Data

`2026-05-26`

---

## Autores

- us4-core

---

## Contexto

`RouterLoadBalanceLoss(logits, k)` em `runtime/moe/router_metrics.cpp` calcula
uma perda L2 entre a distribuição de roteamento (softmax dos logits) e uma carga
"ideal" por expert. A implementação usava `idealLoad = k / N`, onde `N` é o
número de experts e `k` o número selecionado no top-k.

O softmax é uma distribuição de probabilidade: soma 1 sobre os `N` experts.
Comparar cada probabilidade (no máximo 1) contra `k / N` é dimensionalmente
inconsistente — para qualquer `k > 1` o alvo passa de `1/N` e a perda fica
positiva mesmo quando o roteador é perfeitamente uniforme. Um roteador uniforme
é, por definição, o caso de balanceamento perfeito e deveria produzir perda
zero.

O sintoma apareceu quando a suíte de contrato passou a compilar de fato (ver
provisionamento de GTest via FetchContent): o teste
`MoeExtendedContractTest.LoadBalanceLossIsZeroWhenUniformAndKMatchesExperts`
falhava porque a implementação retornava `2.25` para logits uniformes com
`k = N = 4`, em vez de `0`.

ADRs relacionados: [ADR-006](./ADR-006-moe-expert-paging.md) (roteamento e
paginação de experts).

---

## Decisão

Adotamos `idealLoad = 1 / N` como alvo de carga por expert no
`RouterLoadBalanceLoss`.

- Escopo: somente a constante de alvo dentro de `RouterLoadBalanceLoss`. A
  fórmula da perda (soma dos quadrados dos desvios em relação ao alvo) e a
  assinatura pública permanecem.
- Como aplicar: substituir `static_cast<float>(k) / N` por `1.0F / N`. O
  parâmetro `k` continua na assinatura por compatibilidade e para uso futuro,
  mas não entra mais no alvo.
- Dono: us4-core (camada MoE).

---

## Consequências

### Positivas (+)

- Roteador uniforme produz perda zero, como esperado por qualquer métrica de
  load-balance de MoE.
- A telemetria `loadBalanceLoss` exposta por `ComputeRoutingTelemetry` passa a
  ser comparável entre valores de `k`.
- Remove um valor de correctness silenciosamente errado que nunca rodou no CI.

### Negativas (-)

- Mudança de valor numérico observável em `loadBalanceLoss`. Nenhum consumidor
  de produção depende dele hoje (os únicos chamadores eram os testes), então o
  raio de impacto é nulo.

### Neutras / observações

- `RouterDecision.loadBalance` (em `runtime/moe/router.cpp`) é um campo distinto
  e não foi afetado por esta decisão.

---

## Alternativas consideradas

### Alternativa A — Manter `k/N` e ajustar o teste

- Resumo: aceitar a implementação atual como contrato e mudar a expectativa do
  teste para `2.25`.
- Por que foi descartada: cimentaria um alvo dimensionalmente incorreto; a perda
  deixaria de ter a propriedade básica "uniforme => zero".

### Alternativa B — Normalizar probs por `k` antes de comparar

- Resumo: escalar a distribuição para somar `k` e manter `idealLoad = k/N`.
- Por que foi descartada: muda a semântica do softmax do roteador sem ganho; o
  alvo `1/N` é a formulação padrão e mais simples.

---

## Critério de revisão

- Se for introduzida uma definição de load-balance baseada em fração de tokens
  despachados (estilo Switch Transformer `N * sum(f_i * P_i)`), reabrir para
  alinhar a métrica ao novo modelo.

---

## Links

- Issue / task: análise de cobertura de testes (branch `claude/test-coverage-analysis`)
- Documentos relacionados: [DESIGN](./DESIGN.md), [PATTERNS](./PATTERNS.md)
- ADRs relacionados: [ADR-006](./ADR-006-moe-expert-paging.md)
