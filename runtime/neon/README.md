# NEON

Home for NEON and Accelerate-based CPU fallback paths.

What already exists:

- `neon_matmul.cpp` and `neon_attention.cpp` still bridge to the scalar
  reference path, which keeps regression behavior stable
- `kernel_profile.{h,cpp}` now describes the intended SIMD shape for matmul and
  attention (`fp32-lane4`, `fp16-lane8`, `bf16-lane8`, `int8-dot`)
- `dequant_int8.{h,cpp}` and `dequant_int4.{h,cpp}` now implement the
  group-wise dequant contracts used by low-bit NEON planning
- the NEON profile layer already locks tile expectations such as `8x8`,
  `4x16`, dot-product usage, and fused softmax-rescale intent
- the backend selector now also checks `neon_vector_bits` and CPU cluster
  eligibility before it chooses NEON automatically
- `neon_matmul.cpp` now has a dedicated fp32 `1x4` microkernel path and uses
  `arm_neon.h` lane-4 vector math on ARM hosts while keeping the same
  contractual fallback elsewhere
- `neon_matmul.cpp` now also executes `fp16` and `bf16` inputs by decoding
  16-bit storage and accumulating into `fp32`, which finally makes the
  advertised `fp16-lane8` / `bf16-lane8` planning surface executable
- `neon_attention.cpp` now has its first real fp32 NEON dot-product path for
  rank-2 attention, still preserving scalar fallback off ARM plus causal/cache
  contract compatibility
- that attention path now also normalizes scores once per row and accumulates
  `value` on lane-4 vectors before tail-scalar cleanup when `valueWidth` is not
  a multiple of 4

Backend implementation gaps:

- real ARM intrinsics and Accelerate hot paths behind those profiles
- fused or intrinsic-backed dequantization kernels for INT8 and INT4

Low-bit observability that already exists:

- `benchmarks/dense_baseline.cpp` now runs the low-bit fixtures in paired
  `scalar` and `neon` passes
- each low-bit case emits `requested_backend`, `observed_backend`,
  `backend_reason`, `fell_back`, `dequant_path`, `neon_kernel_flavor`,
  `elapsed_ms`, and a stable `text_fingerprint`
- each pair also emits a machine-readable `regression_status` plus
  `regression_reason` so correctness drift and silent fallback do not hide in
  the bench output
- `lowbit-int8` currently expects `groupwise-int8` plus `int8-dot`
- `lowbit-int4` currently expects `groupwise-int4`; `scalar-bridge` is still a
  valid visible kernel flavor until a true intrinsic path lands

Observability gaps:

- dedicated correctness binaries under `runtime/benchmarks/correctness/`
- Apple-host benchmark tables with committed Sprint 04 numbers
