# Migration Guide

> Origem: Sprint 12 / T12.7. Migration + troubleshooting guide para usuarios
> que sobem de pre-release para v1.0.

## 1. Escopo

- Migracao do contrato CLI pre-v1.0 para o novo command surface.
- Migracao do contrato JSON pre-v1.0 para o schema v1.
- Lista de troubleshooting comum em hosts Apple Silicon.

## 2. CLI command migration

| Antes | Depois |
|---|---|
| `us4-cli --probe` | `us4-cli probe` (legado continua funcionando ate v1.1) |
| `us4-cli --mode auto` | `us4-cli probe --mode auto` |
| `us4-cli run --model X --prompt Y` | mesma assinatura |
| `us4-cli list-models` | sem mudanca |

Apos v1.1 o `--probe` legado emite warning explicito no `stderr`.

## 3. JSON schema migration

Pre-v1.0 expunha `probe` e `mode` como blocos top-level. v1.0 envolve tudo
em `result.telemetry`. Veja `.specs/runtime/CLI-POLISH.md` para o schema.

Para o consumo automatizado existente:

```bash
us4-cli probe --json | jq '.probe // .result.probe'
```

## 4. Troubleshooting

### Backend `metal` reporta `fell_back=true`

Causa comum: chip nao expoe Metal device (CI Linux, x86_64). O runtime
degrada para `mlx` ou `scalar` automaticamente. Verifique
`backend_reason`.

### Backend `ane` reporta `chip-too-old`

Chip abaixo de M5. Sem fallback transparente: o runtime mantem Metal.
Veja `runtime/ane/README.md`.

### `kv` telemetry mostra hit ratio baixo

Confira o modo selecionado. `MICRO`/`NANO` evictam mais agressivamente.
Force `--mode FULL` para diagnostico.

### Speculative decoding nao aplica

Cheque se `--speculative` foi solicitado e se o draft model esta presente
no manifest. `speculative.draft_attempts=0` indica que o caminho nao
disparou.

### `playwright test` falha sem evidencia

Cheque `playwright-report/index.html` + `test-results/`. O gate DoD
bloqueia merge sem essa evidencia.

## 5. Referencias

- `.specs/runtime/CLI-POLISH.md`
- `.specs/workflow/RELEASE-V1.md`
- `.specs/workflow/WORKFLOW.md`
