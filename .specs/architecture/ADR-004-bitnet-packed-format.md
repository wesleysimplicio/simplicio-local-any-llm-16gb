# ADR-004: BitNet and Ternary Packed Format

- Status: Accepted
- Date: 2026-05-18
- Owners: us4-core

## Context

Sprint 05 introduces BitNet 1.58-bit and PT-BitNet ternary adapters. Both
adapters represent weights using the same {-1, 0, +1} alphabet, so the runtime
needs one stable on-disk and in-memory packed format that the Metal and NEON
kernels can decode without ambiguity.

## Decision

Ternary weights are encoded using base-3 packing:

- BitNet packed byte: 5 ternary values per byte. The least-significant ternit
  occupies digit 0; the most-significant occupies digit 4. Valid byte values
  are `[0, 243)`; bytes `[243, 256)` are reserved.
- Ternary LUT chunk: 4 ternary values per byte (same base-3 ordering). The
  lookup table has exactly 81 entries (`3^4`). Bytes `[81, 256)` are reserved.

Per-row dequantization scales are stored alongside the packed payload. The
runtime applies the scale after the integer-domain dot product, so the
kernels stay bit-exact regardless of float precision.

Loader contract for BitNet GGUF / ternary safetensors:

- BitNet GGUF assets advertise `weight_format=bitnet-b1.58` plus a `row_scale`
  field in the sibling manifest.
- Ternary safetensors assets advertise `weight_format=pt-bitnet-ternary` and
  reuse the same `row_scale` field.
- The loader rejects any asset whose declared `weight_format` does not match
  the adapter family while preserving the visible loader telemetry.

## Consequences

- `runtime/neon/bitnet_matmul.{h,cpp}` provides reference encoding/decoding,
  matmul, and per-row scaling. The packed-vs-unpacked outputs stay
  parity-checked via `tests/unit/lowbit_contract_test.cpp`.
- `runtime/cpu/ternary_lut.{h,cpp}` materializes the 81-entry LUT for 4-chunk
  ternary dot products and is the canonical reference for any future Metal/
  NEON ternary path.
- `runtime/metal/kernels/bitnet_matmul.metal` documents the packed decoding
  contract on the GPU side; the actual kernel compilation lands together with
  the Metal toolchain integration.
- MICRO and MICRO_PLUS modes prefer BitNet/Ternary adapters when the runtime
  asks for a low-memory route; this is enforced through the registry and the
  `RuntimeModeSelector` (see Sprint 05 T05.7).

## Alternatives considered

- Bit-packed 2-bit-per-weight format: wastes a state out of the four (we only
  need three) and requires an explicit "tombstone" handling rule.
- 1-byte-per-weight format: trivial to decode but defeats the memory savings
  that BitNet and Ternary adapters need to deliver in MICRO mode.
