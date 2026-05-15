---
sprint: sprint-02
status: in_progress
start: 2026-05-28
end: 2026-06-10
owner: us4-core
---

# Sprint 02 - CPU Scalar Baseline (Apple)

## Objetivo
Tensor primitives + baseline escalar (matmul, attention) em C++ puro. Adapter dense base com Qwen + Gemma rodando 100% CPU como referencia para correctness diff.

## Tasks
- [x] T02.1 - `runtime/core/tensor.{h,cpp}` (shape, dtype FP32/FP16/BF16/INT8/INT4, stride, device tag)
- [x] T02.2 - `runtime/cpu/scalar_matmul.cpp` (reference scalar GEMM, FP32)
- [x] T02.3 - `runtime/cpu/scalar_attention.cpp` (causal mask, softmax stable, KV concat)
- [x] T02.4 - `runtime/adapters/DenseAdapterBase` (shared dense transformer loop)
- [x] T02.5 - `runtime/adapters/qwen/QwenAdapter` (config, tokenizer, scalar path)
- [x] T02.6 - `runtime/adapters/gemma/GemmaAdapter` (config, scalar path)
- [x] T02.7 - Loader GGUF/Safetensors minimal (fixture manifest + format detection para assets FP32/FP16 nesta sprint)
- [x] T02.8 - Benchmark harness `runtime/benchmarks/dense_baseline.cpp`

## Test plan
- Unit: matmul vs naive Python reference (atol 1e-4); attention output shape/mask; tokenizer round-trip.
- Regression: Sprint 01 probe + mode selector still green.
- E2E: `us4-cli run --model qwen-0.5b --prompt "hi"` gera >= 5 tokens em <=60s CPU.
- Correctness: logit diff vs reference dentro de 1e-3 nos primeiros 32 tokens.

## DoD
- 2 adapters gerando texto coerente em CPU scalar.
- Coverage >=80% em `runtime/cpu` + `runtime/adapters/{qwen,gemma}`.
- Bench numbers logados.

## Status atual
- O caminho nativo agora cobre `run --model ... --prompt ...` com geracao deterministica em CPU scalar.
- Fixtures locais existem em `tests/fixtures/models/` para Qwen e Gemma.
- O benchmark baseline ja executa o adapter Qwen no caminho scalar.
- Validacao nativa completa segue bloqueada neste host por ausencia de toolchain C++/CMake no `PATH`; o gate real continua em CI/macOS ate essa parte do ambiente existir localmente.

## Riscos
- GGUF loader edge cases (quantizacoes raras) -> defer pro Sprint 04.
