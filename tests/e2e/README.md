# E2E smoke

Esta pasta agora cobre o estado real do repo: o starter CLI em JavaScript.

## Escopo atual

- alvo executado: `node bin/cli.js`
- contrato smoke: `--version`, `--probe` e `--mode auto --json`
- runner: Playwright Test como orquestrador e produtor de evidencias

## O que o smoke valida

- `--version` responde em texto puro e em JSON
- `--probe` responde em texto com resumo de modo e hardware
- `--probe --json` e `--mode auto --json` expoem o contrato estruturado esperado
- cada execucao anexa stdout/stderr no report do Playwright

## Evidencia esperada

- `playwright-report/index.html`
- `test-results/results.json`
- `test-results/results.xml`
- attachments `stdout-*` e `stderr-*` por caso dentro do JSON report
- pelo menos um attachment `trace` apontando para `trace.zip`

## Gate de evidencia

Os workflows de CI/DoD tratam a evidencia do Playwright como contrato:

- falha se `playwright-report/index.html` nao existir
- falha se `test-results/results.{json,xml}` nao existirem
- falha se o JSON report perder os attachments `stdout-*` / `stderr-*`
- falha se nenhum attachment `trace` referenciar `trace.zip`

## Nota de transicao

O nome de contrato continua `us4-cli`, mas o binario real hoje ainda e o starter JS.
Quando o CLI nativo existir, a migracao esperada aqui e trocar o comando executado
no helper do spec, preservando as mesmas assercoes de contrato onde fizer sentido.
