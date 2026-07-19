# Definition of Done — property/fuzzing layer

Complements (does not replace) the DoD already in `CLAUDE.md`/`AGENTS.md` --
build/lint/unit/E2E/regression, and in particular the **hard correctness
gate**: "Correctness > performance. Logit diff vs referência é gate duro"
(`runtime/benchmarks/correctness/correctness_check.cpp`, tolerance `1e-4`
against the scalar CPU reference for matmul/RoPE). This file adds the 4th
DoD layer tracked by the ecosystem-wide hub issue
([simplicio-loop#579](https://github.com/wesleysimplicio/simplicio-loop/issues/579))
for this repo
([simplicio-local-any-llm-16gb#144](https://github.com/wesleysimplicio/simplicio-local-any-llm-16gb/issues/144)):
property/fuzz coverage for the native kernels, on top of the existing
example-based GoogleTest contract suite (`tests/unit/*_contract_test.cpp`).

## What already exists (do not duplicate)

`runtime/fuzz/` already ships libFuzzer harnesses, opt-in via
`-DUS4_BUILD_FUZZERS=ON` (combine with `-DUS4_ENABLE_ASAN=ON
-DUS4_ENABLE_UBSAN=ON`, Clang required for `-fsanitize=fuzzer`):

- `fuzz_json_value.cpp`
- `fuzz_safetensors_reader.cpp`
- `fuzz_gguf_reader.cpp`
- `fuzz_bpe_tokenizer.cpp` -- **tokenizer fuzzing is already covered.**

These target the file-format parsers (untrusted bytes from a downloaded
model/tokenizer file, see `runtime/fuzz/CMakeLists.txt` header comment
referencing #81.13). What is **not yet covered** is the quantized-kernel
math itself.

## New: property/fuzzing layer for the quantization kernels

- **Tool: AFL++ (primary) + libFuzzer (secondary, reuses the existing
  `runtime/fuzz/` harness convention)**. AFL++ for the standalone
  corpus-driven fuzzing campaign (real weight files as seed corpus, per the
  follow-up issue), libFuzzer for CI-integrated quick regression fuzzing
  alongside the existing four harnesses.
- **Target kernels**: `runtime/neon/dequant_int8.{h,cpp}`,
  `runtime/neon/dequant_int4.{h,cpp}` (the NEON dequantization kernels named
  explicitly in this repo's own DoD-review ask), plus
  `runtime/cpu/quantize_projection.{h,cpp}` as the scalar reference they must
  agree with.
- **Invariant, not just "doesn't crash"**: for arbitrary (including
  adversarial/corrupted) quantized-block byte input, `dequant_int8`/
  `dequant_int4` must either (a) reject the input via the kernel's own
  documented error/bounds contract, or (b) produce output that matches the
  scalar dequantization reference within the same `1e-4` logit-diff-class
  tolerance already codified in `correctness_check.cpp` -- reusing that
  tolerance instead of inventing a new one keeps this layer consistent with
  the existing hard correctness gate rather than adding a second, competing
  notion of "close enough."
- **Corpus discipline**: seed the fuzzer/AFL++ corpus with **slices of real
  GGUF/safetensors quantized weight blocks** (e.g. from a small real
  `qwen-0.5b`-class quantized model already used elsewhere in this repo's
  benchmarks), not only synthetic all-zero/all-`0xFF` blocks -- synthetic-only
  corpora systematically miss the value distributions a real quantizer
  actually produces (clustered scale/zero-point patterns, not uniform noise).
  Step-by-step plan and corpus source: tracked in the follow-up issue this
  file's commit references.

## Cross-backend invariant review (MLX / Metal / NEON indexing agreement)

Separate from fuzzing: a **structural review question** this repo's DoD did
not previously ask explicitly, added here because a divergence here would
silently corrupt every model that routes through more than one backend
(`runtime/mlx/`, `runtime/metal/`, `runtime/neon/`, dispatched per
`ADR-003-backend-selection-strategy.md`):

> **Do the three backends that process the same tensor structure
> (`runtime/mlx/dense_plan.{h,cpp}` + `mlx_bridge.{h,cpp}`,
> `runtime/metal/dense_dispatch.{h,cpp}`, `runtime/neon/block_gemm.{h,cpp}` +
> `neon_matmul.{h,cpp}`) agree on the same indexing/stride convention for a
> tensor (row-major vs. column-major, `Tensor::strides_` in
> `runtime/core/tensor.h`) at every boundary where a tensor crosses from one
> backend's dispatch path into another's?**

This is exactly the kind of bug `correctness_check.cpp` is designed to catch
*if* the test matrix already exercises a cross-backend handoff -- today the
comparison there is "scalar vs NEON, and on Apple the dispatched Metal/MLX
simulation" per its own header comment, i.e. it may already implicitly cover
this. This DoD item makes it an **explicit, named review question** any
change to the three backends' tensor-handling code must answer in the PR
(see the PR template checkbox below), rather than something covered only
incidentally by whichever correctness cases happen to already exist. It does
not mandate new test infrastructure by itself -- if a reviewer answers "no,
not covered for this change," that answer must come with either a new
correctness-gate case or an explicit, reviewed risk acceptance, never a
silent skip.

## Validation performed for this change

This repo is C++/CMake/Metal/MLX, targeting Apple Silicon
(`macos-14` CI runner per `AGENTS.md`). **No `cmake`/build was attempted in
this (Linux) environment** -- it would not succeed regardless of this
change (Metal/MLX/ANE are Apple-only), so a fake "build passed" result would
violate this repo's own "sem mockar/fake pass" rule. What was actually run:

```
$ npm run lint
node --check bin/cli.js && node --check bin/us4-cli.js && node --check test/run-tests.js \
  && node --check test/cli.test.js && node --check test/us4-cli.test.js
(no output -- all files parse clean)

$ npm test
# tests 18
# pass 18
# fail 0
```

Those are the only auxiliary scripts in this repo that run without the Apple
toolchain (`bin/cli.js`/`bin/us4-cli.js` + their Node `--test` suite in
`test/`, driven by `test/run-tests.js`). `scripts/coverage.sh` requires a
full `cmake`/Ninja/Clang build; `scripts/openai_serve.py` requires `mlx`
(Apple-only wheel, per `scripts/requirements-serve.txt`); neither was run.
`cmake --build`, `ctest`, `clang-format`/`clang-tidy`, and
`npx playwright test` (per `AGENTS.md`'s workflow loop) were **not** run
here and must be run on real Apple Silicon hardware/CI before merge.
