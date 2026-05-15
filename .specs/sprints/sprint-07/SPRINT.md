---
sprint: sprint-07
status: todo
start: 2026-08-06
end: 2026-08-19
owner: us4-core
---

#Sprint 07 - Llama Adapter(Apple)

## Objetivo
Adapter Llama 3/4 com GQA, RoPE scaling (linear/dynamic/YaRN), ALiBi opcional. Full attention path em todos backends ja existentes.

## Estado atual no repo em 2026-05-15
- `LlamaAdapter` ja esta no registry nativo e aparece no CLI.
- `LlamaConfig` ja resolve `hidden_size`, `query_heads`, `kv_heads`,
  `head_dim`, `rope_theta`, `rope_scaling` e `rope_scale` a partir do
  manifesto fixture, com cobertura de normalizacao segura.
- O adapter ja possui um caminho dedicado para `neon`, com KV reutilizavel por
  prompt/seed no mesmo `RuntimeContext`, enquanto os demais backends ainda
  dependem do shared path do `DenseAdapterBase`.
- `runtime/core/rope.{
  h, cpp
}
` ja saiu do placeholder e agora expõe um contrato deterministico para scaling `linear`, `dynamic` e `YaRN` no
                                                                                                 layout 2D atual
                                                                                                     .-
                                                                                             Loader
                                                                                                 real,
    corretude contra referencia e execucao Llama nos backends continuam como
            meta do sprint abaixo.

        ##Tasks -
        [x] T07.1 - `runtime / adapters / llama /
                        LlamaConfig` (rope_theta, rope_scaling, gqa heads) -
        [] T07.2 - `runtime / adapters / llama
                       / LlamaAdapter` (forward pass, KV via pager) -
        [x] T07.3 - `runtime / core / rope.{
  h, cpp
}
` (linear + dynamic + YaRN scaling) -
    [] T07.4 - `runtime / core / gqa_attention.{
  h, cpp
}
` (group query attention) -
    [] T07.5 - Loader : Llama GGUF + safetensors + tokenizer.json -
                        [] T07.6 - Bench Llama 3.x 8B em Metal +
                        NEON

                        ##Test plan
                        -
                        Unit : RoPE vs reference Python(atol 1e-5); GQA shape contract.
- Regression: outros adapters intactos.
- E2E: Llama 3 8B Q4 em M3 Max gera 200 tokens em <= 30s.
- Correctness: diff vs HF reference <= 1e-3 nos primeiros 64 tokens.

## Contract prep before implementation
- Unit contract should keep Llama fixture coverage for:
  - manifest directory loading without explicit `--model`
  - default prompt token fallback from `model.us4manifest`
  - GGUF asset detection and family/model routing
  - requested backend fallback telemetry when `metal` is unavailable
- Native E2E should keep host-aware Llama evidence for both:
  - `tests/fixtures/models/llama-3.1-8b/`
  - `tests/fixtures/models/llama-3.1-8b/toy-llama.gguf`
- Bench evidence for T07.6 should record at minimum:
  - requested backend, observed backend, and fallback reason
  - runtime mode and hardware profile
  - generated token count, elapsed time, and text fingerprint
  - correctness delta once the HF reference path exists

## DoD
- Llama 3 + 4 funcionando em FULL/BALANCED_PLUS/DEGRADED.
- Coverage >=80% em `runtime/adapters/llama`.
