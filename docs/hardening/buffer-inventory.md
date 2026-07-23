# Untrusted-input buffer inventory

This inventory is the review receipt for issue #122. It separates buffers whose
size is derived from a model or request from small fixed buffers whose contract
is independent of tensor dimensions.

| Path and buffer | Input that can pressure it | Decision | Evidence/status |
| --- | --- | --- | --- |
| `engine/c/glm.c`, `rope_interleave` input copy | `qk_rope_head_dim` | Allocate with `falloc(qk_rope)` | **Fixed and verified** (compiled, ASan/UBSan clean, GLM+DeepSeek TF 32/32). The `CKR` gate additionally caps `qk_rope_head_dim` at 256 as defense-in-depth. |
| `engine/c/glm.c`, DSA `w32` | `index_n_heads` | Allocate with `falloc(index_n_heads)` (or `nh`) | **Fixed and verified**; `index_n_heads` is validated by `CKR` (tightened to `[0,64]`). |
| `engine/c/glm.c`, absorbed attention `qabs`/`clat` | `kv_lora_rank` | Allocate with `falloc(kv_lora_rank)` | **Fixed and verified**; the path is entered only after the existing rank checks. |
| `engine/c/glm.c`, attention scores (both the MLA-absorption and the full-reconstruction path) | context length and DSA selection count | Allocate **one bulk buffer per `attention()` call**, sized `S*H*Tk`, sliced per `(s,h)` OpenMP work item | **Fixed and verified.** Revised from an initial per-iteration `falloc`/`free` inside the `#pragma omp parallel for` loop to a single allocation before the parallel region — same safety guarantee, avoids malloc/free churn across thousands of iterations in the hot decode path. |
| `engine/c/glm.c`, `forward_all` final norm row | `hidden_size` | Reuse one heap row sized by `hidden_size` | **Fixed and verified**; the old `row[8192]` was not consistent with the config limit. |
| `engine/c/olmoe.c`, attention scores | context length | Allocate one bulk score buffer (`H*S*Tk`) before the parallel loop, sliced per `(hh,s)` | **Fixed and verified**; same bulk-allocation revision as above (was per-iteration `falloc(qpos+1)`/`free`). The old `sc[4096]` was an implicit context ceiling. |
| `engine/c/glm.c`, `rope_interleave` reading the **caller's** vector (not just its own internal copy) | `index_head_dim` vs `qk_rope_head_dim` when the DSA indexer calls `rope_interleave` on its `index_head_dim`-sized `k_idx` buffer | New `CKR`-adjacent invariant: reject configs where `index_head_dim < qk_rope_head_dim` | **New finding, fixed and verified.** Heap-allocating `rope_interleave`'s own `in[]` copy (sized by `qk_rope_head_dim`) does not fix this: the function still `memcpy`s/writes `qk_rope_head_dim` floats into/from the **caller's** vector, which is only `index_head_dim` floats wide when called from the DSA indexer path. If `index_head_dim < qk_rope_head_dim`, that is still an out-of-bounds read/write on the caller's buffer regardless of `in[]`'s own sizing. Caught during merge reconciliation of this patch with an earlier follow-up that only fixed `rope_interleave`'s internal buffer. |
| `engine/c/glm.c`, `stop_ids[8]` and `g_stop[9]` | configured EOS/stop list | Keep fixed, validate and cap the parsed list | The parser stores at most eight configured IDs and reserves one slot for the explicit EOS; adversarial-config regression coverage now exercises the config choke point. |
| `engine/c/glm.c`, `draft[64]`/`batch[64]` | speculative window | Keep fixed, clamp `g_draft` to 63 before use | The decode path clamps the user-controlled draft window; adversarial dimension cases are covered by `test_adversarial_config.py`. |
| `engine/c/glm.c`, `ws[64]` and `missk[64]` | active experts per batch | Keep fixed, validate `topk`/working-set limits | These are routing work buffers, not tensor-dimension buffers; `num_experts_per_tok` is now covered by the adversarial config gate. |
| `engine/c/tok.h`, byte maps and UTF-8 scratch | byte alphabet and token encoding | Keep fixed by the byte-level tokenizer contract | The dimensions are protocol constants, independent of vocabulary size. |

## Verification boundary

**Updated.** A follow-up session had a Linux x86_64 host with a working GCC
toolchain and closed the gap the previous receipt flagged as required:

- `make test-c` / `make test-python`: green (3/3 C tests, 53/53 Python tests).
- `make sanitize` (new target, `engine/c/Makefile`): the engine binary and all
  three C test binaries built with `-fsanitize=address,undefined
  -fno-omit-frame-pointer -fno-sanitize-recover=all`, then run against the
  full C test suite plus the GLM and DeepSeek teacher-forcing oracles
  (`TF=1`, 32/32 positions each). Zero AddressSanitizer or
  UndefinedBehaviorSanitizer reports. Leak detection is deliberately disabled
  for this run (`ASAN_OPTIONS=detect_leaks=0`) because `json.h`'s parsed
  `jval` tree is process-lifetime by design across this whole codebase
  (`glm.c`'s own `load_cfg`/`cfg_root` only ever free the raw-text arena,
  never the node tree) — a one-shot config/tokenizer parse in a short-lived
  CLI/server process is not a memory-safety bug, and leak detection would
  only flag that known, intentional pattern instead of the overflow/UB
  corruption this target exists to catch.
- The bulk-allocation buffers above were reconciled from two independent
  fixes for the same root cause (this session's and an earlier follow-up's)
  during a rebase; the merge was re-validated with a full clean build + test
  run + both TF oracles afterward, not assumed correct from either side.

**Still open** (tracked by issue #122, not closed by this patch): the four
libFuzzer harnesses (`json.h`, `st.h`, `tok.h`, server line protocol) and
full generated-oracle sanitizer run. The adversarial-config suite is now
committed as `engine/c/tests/test_adversarial_config.py`; it executes the real
engine binary against hostile dimensions and asserts an ordinary, actionable
rejection before tensor loading. `make sanitize` proves the buffers found by
manual audit are safe under the inputs exercised by the existing test/oracle
suite; it does not replace coverage-guided fuzzing for inputs outside that
set.
