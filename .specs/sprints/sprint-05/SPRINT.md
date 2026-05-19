---
sprint: sprint-05
status: done
start: 2026-07-09
end: 2026-07-22
owner: us4-core
---

# Sprint 05 — BitNet + Ternary Adapters (Apple)

## Objetivo
BitNet 1.58-bit + Ternary (PT-BitNet) com kernels Metal + NEON. MICRO mode rodando em RAM minima.

## Estado atual no repo em 2026-05-14
- `BitNetAdapter` e `TernaryAdapter` ja estao registrados no runtime nativo e aparecem em `us4-cli list-models`.
- Hoje eles ainda se comportam como adapters de scaffold sobre o caminho deterministico compartilhado; kernels packed, loaders reais e comportamento MICRO ainda nao pousaram.
- O escopo abaixo continua sendo a entrega esperada para execucao low-memory de verdade.

## Tasks
- [x] T05.1 — `runtime/metal/kernels/bitnet_matmul.metal` (1.58-bit packed)
- [x] T05.2 — `runtime/neon/bitnet_matmul.cpp` (packed lookup + popcount)
- [x] T05.3 — `runtime/adapters/bitnet/BitNetAdapter` (load packed weights, scale layers)
- [x] T05.4 — `runtime/adapters/ternary/TernaryAdapter` (PT-BitNet ternary -1/0/+1)
- [x] T05.5 — Ternary lookup tables (LUT 256-entry for 4-ternary chunks)
- [x] T05.6 — Loader: BitNet GGUF variant + ternary safetensors
- [x] T05.7 — RuntimeMode MICRO trigger (RAM<=8GB -> ternary preferido)

## Test plan
- Unit: BitNet matmul Metal vs scalar reference (atol 5e-3); ternary LUT correctness.
- Regression: dense adapters (Qwen/Gemma) intactos.
- E2E: `us4-cli run --model bitnet-b1.58-2b --mode micro` gera 64 tokens em <= 30s.
- Correctness: diff vs HF reference <= 1e-2 (BitNet inherently lossy).

## DoD
- BitNet + Ternary funcionando em MICRO + MICRO_PLUS.
- Coverage >=80% nos adapters novos.
- ADR-004 BitNet packed format.
