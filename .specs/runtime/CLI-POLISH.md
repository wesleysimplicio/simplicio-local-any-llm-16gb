# CLI Polish - `us4-cli` Final Surface

> Origem: Sprint 12 / T12.4. Define o comando completo planejado para v1.0,
> formato JSON estavel e plano de cobertura E2E.

## 1. Command surface

A versao v1.0 expoe seis subcomandos canonicos:

| Comando | Resumo |
|---|---|
| `us4-cli probe` | Hardware probe + recommended mode (substitui flag `--probe`) |
| `us4-cli list-models` | Lista adapters registrados |
| `us4-cli run` | Geracao single-shot (`--model`, `--prompt`, `--backend`, `--mode`) |
| `us4-cli serve` | Servidor local que aceita prompts e mantem sessoes |
| `us4-cli bench` | Roda `dense_baseline` e/ou `matrix_runner` |
| `us4-cli tune` | Executa auto-tuner e popula `runtime/tuning/profiles.json` |

Flags compartilhadas: `--json`, `--mode <name>`, `--backend <name>`.

## 2. JSON contract

Todos os subcomandos suportam `--json`. Estrutura comum:

```json
{
  "cli": "us4-cli",
  "version": "<semver>",
  "command": "<run|serve|...>",
  "result": { ... },
  "telemetry": {
    "backend": { "requested": "auto", "observed": "metal", "fell_back": false },
    "mode": { "selected": "FULL", "source": "memory-tier" },
    "kv": { "hit_hot": 0, "hit_warm": 0, "hit_cold": 0, "summarize_ratio": 0.0 },
    "moe": { "expert_hit_rate": 0.0, "router_entropy": 0.0 },
    "speculative": { "draft_attempts": 0, "acceptance_rate": 0.0 }
  }
}
```

Schema versionado em `us4-cli serve --schema`.

## 3. E2E coverage plan

- `tests/e2e/us4-cli.spec.ts` cobre `--version`, `--probe`, `--mode auto`
  e `run --model qwen-0.5b --prompt hi` (single-shot).
- Novos casos planejados para v1.0:
  - `us4-cli probe --json` (substitui `--probe`)
  - `us4-cli run --backend metal --json` em fixtures Llama e Qwen
  - `us4-cli bench --short --json` capturando `regression_status`
  - `us4-cli tune --dry-run --json` validando profile cache write
- Evidence rule: cada execucao anexa `stdout`, `stderr`, `trace` ao report.

## 4. Backward-compat

- `--version`, `--probe`, `--mode auto` continuam funcionando ate v1.0.
- Apos v1.0 o `--probe` legado entra em deprecation com warning explicito.

## 5. Referencias

- `.specs/runtime/CLI-CONTRACT.md`
- `.specs/sprints/sprint-12/SPRINT.md`
- `tests/e2e/us4-cli.spec.ts`
