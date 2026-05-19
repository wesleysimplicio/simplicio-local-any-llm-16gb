# Runtime Specs - Contratos Operacionais

> Esta pasta contem contratos operacionais do runtime que sao **normativos**
> para testes, CI e DoD. Documentos de produto vivem em `.specs/product/` e
> documentos de arquitetura em `.specs/architecture/`.

## Indice

| Documento | Escopo |
|---|---|
| [`CLI-CONTRACT.md`](CLI-CONTRACT.md) | Contrato de output do `us4-cli` (`--version`, `--probe`, `--mode`) em texto e JSON. |
| [`HARDWARE-PROBE.md`](HARDWARE-PROBE.md) | Responsabilidades do `HardwareProbe`, taxonomia de modos e mapeamento memoria-tier. |
| [`TELEMETRY.md`](TELEMETRY.md) | Categorias de telemetria, output e gates por fase de maturidade. |
| [`PLAYWRIGHT-SMOKE.md`](PLAYWRIGHT-SMOKE.md) | Escopo, alvos e evidencias minimas do smoke E2E. |
| [`CLI-POLISH.md`](CLI-POLISH.md) | Surface CLI completa planejada para v1.0 + plano E2E. |

## Regras

- Cada documento aqui pode ser referenciado diretamente em testes Playwright
  e em testes unitarios.
- Mudancas de contrato exigem PR + atualizacao da doc + commit Conventional.
- Mudancas que quebrem schema JSON exigem ADR adicional.
