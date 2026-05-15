# CPU Scalar Reference Path

Sprint 02 introduces `runtime/cpu/` as the portable scalar reference path for dense execution.

- `scalar_matmul.*` is the safe GEMM baseline.
- `scalar_attention.*` is the safe attention baseline.

This directory exists to make correctness work possible before NEON-specific vectorization lands in Sprint 04.
