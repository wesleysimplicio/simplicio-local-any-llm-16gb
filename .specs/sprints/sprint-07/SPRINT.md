---
sprint: sprint-07
status: todo
start: 2026-08-06
end: 2026-08-19
owner: us4-core
---

# Sprint 07 — Llama Adapter (Apple)

## Objetivo
Adapter Llama 3/4 com GQA, RoPE scaling (linear/dynamic/YaRN), ALiBi opcional. Full attention path em todos backends ja existentes.

## Estado atual no repo em 2026-05-14
- `LlamaAdapter` ja esta no registry nativo e aparece no CLI.
- O adapter declara capacidade MLX e ja se ancora em headers como `rope` e `gqa`, mas a geracao ainda delega ao `DenseAdapterBase`.
- Loader real, corretude contra referencia e execucao Llama nos backends continuam como meta do sprint abaixo.

## Tasks
- [ ] T07.1 — `runtime/adapters/llama/LlamaConfig` (rope_theta, rope_scaling, gqa heads)
- [ ] T07.2 — `runtime/adapters/llama/LlamaAdapter` (forward pass, KV via pager)
- [ ] T07.3 — `runtime/core/rope.{h,cpp}` (linear + dynamic + YaRN scaling)
- [ ] T07.4 — `runtime/core/gqa_attention.{h,cpp}` (group query attention)
- [ ] T07.5 — Loader: Llama GGUF + safetensors + tokenizer.json
- [ ] T07.6 — Bench Llama 3.x 8B em Metal + NEON

## Test plan
- Unit: RoPE vs reference Python (atol 1e-5); GQA shape contract.
- Regression: outros adapters intactos.
- E2E: Llama 3 8B Q4 em M3 Max gera 200 tokens em <= 30s.
- Correctness: diff vs HF reference <= 1e-3 nos primeiros 64 tokens.

## DoD
- Llama 3 + 4 funcionando em FULL/BALANCED_PLUS/DEGRADED.
- Coverage >=80% em `runtime/adapters/llama`.
