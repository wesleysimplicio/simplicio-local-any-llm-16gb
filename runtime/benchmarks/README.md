# Benchmarks

Este diretorio guarda harnesses de benchmark e os contratos de observabilidade
que sustentam comparacoes honestas entre backends.

## Estado atual

- `dense_baseline.cpp` mede o scaffold denso atual e imprime metadados de
  backend, fallback, token count, tempo e fingerprint do texto gerado.
- Os numeros atuais ainda sao evidencia de contrato, nao prova de throughput
  real de Llama 3/4.

## Evidencia minima para o vertical Llama

Quando o benchmark especifico de Llama chegar em `T07.6`, cada execucao deve
registrar pelo menos:

- `benchmark`
- `model`
- `asset_format`
- `requested_backend`
- `observed_backend`
- `backend_reason`
- `generated_tokens`
- `elapsed_ms`
- `text_fingerprint`
- `recommended_mode` ou `mode`
- perfil de hardware relevante para comparar runs

Se a trilha de corretude contra referencia HF ja existir, adicionar tambem o
delta dos primeiros 64 tokens no mesmo output ou em um artefato vizinho.

## Regras

- Nao transformar numero de fixture em claim de performance.
- Nao esconder fallback.
- Nao comparar runs sem hardware profile.
- Nao marcar um backend como suportado so porque o adapter apareceu no registry.

## Comandos uteis

```bash
./build/runtime/benchmarks/dense_baseline
./build/apps/us4-cli run --model-path tests/fixtures/models/llama-3.1-8b --prompt "hello" --json
./build/apps/us4-cli run --model-path tests/fixtures/models/llama-3.1-8b/toy-llama.gguf --prompt "hello" --json
```
