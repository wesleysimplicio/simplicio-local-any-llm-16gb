# Preparação do baseline real em host 16 GB (#118)

Este arquivo existe para impedir “baseline de 16 GB” falso.

## O que precisa ser verdade antes da captura

- A máquina usada para a medição precisa realmente aparentar 16 GB de RAM
  física dentro da tolerância do template.
- O checkpoint real precisa existir no disco local e o caminho precisa ser
  preenchido no template, sem alias inventado.
- O perfil `16gb` precisa ser o mesmo documentado em
  `docs/profiles/16gb.md`.
- O comando capturado precisa ser o mesmo registrado no artefato JSON.

## O que NÃO conta

- Rodar em host com 24 GB, 32 GB ou mais e depois “normalizar” a interpretação.
- Rodar só fixture sintética e promover o número a baseline real.
- Colar manualmente `tok/s`, `rss` ou `perplexity` em markdown.
- Substituir a ausência de hardware por extrapolação do breakdown do residente.

## Procedimento mínimo

1. Ajustar `glm_snap_path` no template
   `docs/benchmarks/fixtures/issue-118-126.template.json`.
2. Gerar um plano:

```bash
python runtime/benchmarks/repro_harness.py run \
  --template docs/benchmarks/fixtures/issue-118-126.template.json \
  --output out/issue-118-126.plan.json \
  --dry-run
```

3. Validar o plano:

```bash
python runtime/benchmarks/repro_harness.py validate \
  --input out/issue-118-126.plan.json
```

4. Rodar a captura real no host 16 GB:

```bash
python runtime/benchmarks/repro_harness.py run \
  --template docs/benchmarks/fixtures/issue-118-126.template.json \
  --output out/issue-118-126.capture.json
```

5. Anexar o JSON bruto à issue/PR/handoff, não só um resumo textual.

## Interpretação honesta

- Se o caso `glm52-int4-16gb-host-baseline` vier como
  `blocked_host_requirement`, #118 continua bloqueada.
- Se o caso executar mas o checkpoint ainda for fixture/local, #118 continua
  bloqueada.
- Se `eval-glm-real-multiple-choice` não rodar por falta de dataset ou
  dependência, #126 continua parcial mesmo que os benchmarks de contrato
  estejam verdes.
