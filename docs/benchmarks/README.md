# Benchmarks reproduzíveis (#126) e baseline 16 GB honesto (#118)

Este diretório junta o que faltava para medir sem inventar:

- um harness de captura em `runtime/benchmarks/repro_harness.py`;
- um template versionado em `docs/benchmarks/fixtures/issue-118-126.template.json`;
- um schema formal do artefato de saída em `docs/benchmarks/repro-bench-run.schema.json`;
- documentação explícita do que é:
  - contrato/fixture local;
  - benchmark real em host de 16 GB;
  - quality eval/perplexity com checkpoint e datasets externos.

## Princípios

- Não preencher números à mão.
- Não chamar fixture de throughput real.
- Não chamar host diferente de 16 GB de “baseline 16 GB”.
- Não esconder ausência de checkpoint, dataset, tokenizer ou binário.
- Não fechar #118 ou #126 com `planned`/`blocked_host_requirement`; isso só prova preparação.

## O que o harness faz

O script `runtime/benchmarks/repro_harness.py`:

- detecta facts do host via stdlib;
- verifica se o host realmente parece 16 GB dentro da tolerância declarada;
- roda os comandos do template quando o host atende aos requisitos;
- captura stdout/stderr/exit code/duração;
- faz parse simples de linhas `chave=valor`;
- grava um JSON auditável;
- valida o formato sem depender de biblioteca externa.

## Fluxos recomendados

### 1. Preparar a captura sem rodar nada

```bash
python runtime/benchmarks/repro_harness.py run \
  --template docs/benchmarks/fixtures/issue-118-126.template.json \
  --output out/issue-118-126.plan.json \
  --dry-run
python runtime/benchmarks/repro_harness.py validate \
  --input out/issue-118-126.plan.json
```

Isso serve para provar que o template, schema e gating estão consistentes,
mesmo em host que não seja 16 GB.

### 2. Medir benchmarks de contrato/fixture

```bash
python runtime/benchmarks/repro_harness.py run \
  --template docs/benchmarks/fixtures/issue-118-126.template.json \
  --output out/issue-118-126.capture.json
```

Em host não-16 GB, os casos que exigem 16 GB ficam com
`status=blocked_host_requirement`. Isso é o comportamento correto.

### 3. Rodar perplexity/evals reproduzíveis

O repo já traz utilitários reaproveitáveis em `engine/c/tools/`:

- `fetch_benchmarks.py` baixa e normaliza datasets para JSONL offline;
- `eval_glm.py` mede accuracy/acc_norm por log-likelihood múltipla escolha.

O template já aponta para esse fluxo, mas ele continua dependente de:

- checkpoint real preparado em disco;
- `tokenizers` instalado;
- datasets baixados localmente;
- host desbloqueado para o experimento que se quer medir.

## Leitura dos status

| status | significado |
|---|---|
| `planned` | template válido, sem execução |
| `captured` | comando executou e gerou evidência |
| `failed` | comando executou e falhou; stdout/stderr preservados |
| `blocked_host_requirement` | host atual não atende o requisito declarado |
| `awaiting_manual_capture` | template sem comando para esta plataforma/etapa |

## O que conta como evidência suficiente

Para #118:

- template + schema + host gate funcionando;
- checklist de preparação do host 16 GB;
- artefato JSON mostrando bloqueio honesto quando o host não atende;
- captura real posterior em máquina 16 GB usando o mesmo template.

Para #126:

- metodologia versionada;
- comandos estáveis e repetíveis;
- artefato JSON versionável;
- distinção clara entre benchmark de contrato, benchmark real e eval de qualidade.

## Não confundir

- `runtime/benchmarks/dense_baseline` e `real_forward_throughput` continuam úteis,
  mas por si só não fecham #118.
- `engine/c/tools/eval_glm.py --dry` prova plumbing, não prova qualidade do modelo.
- `docs/baselines/glm52-dense-breakdown.md` descreve composição do residente;
  não substitui medição física em host 16 GB.
