# Untrusted-input buffer inventory

This inventory is the review receipt for issue #122. It separates buffers whose
size is derived from a model or request from small fixed buffers whose contract
is independent of tensor dimensions.

| Path and buffer | Input that can pressure it | Decision | Evidence/status |
| --- | --- | --- | --- |
| `engine/c/glm.c`, `rope_interleave` input copy | `qk_rope_head_dim` | Allocate with `falloc(qk_rope)` | Fixed in the follow-up patch; the existing config gate still rejects non-positive and oversized dimensions. |
| `engine/c/glm.c`, DSA `w32` | `index_n_heads` | Allocate with `falloc(index_n_heads)` | Fixed in the follow-up patch; `index_n_heads` is validated by `CKR`. |
| `engine/c/glm.c`, absorbed attention `qabs`/`clat` | `kv_lora_rank` | Allocate with `falloc(kv_lora_rank)` | Fixed in the follow-up patch; the path is entered only after the existing rank checks. |
| `engine/c/glm.c`, attention scores | context length and DSA selection count | Allocate one score vector per OpenMP work item | Fixed in the follow-up patch; size is `nt`, the selected key count for that item. |
| `engine/c/glm.c`, `forward_all` final norm row | `hidden_size` | Reuse one heap row sized by `hidden_size` | Fixed in the follow-up patch; the old `row[8192]` was not consistent with the config limit. |
| `engine/c/olmoe.c`, attention scores | context length | Allocate one heap score vector sized by `qpos + 1` | Fixed in the follow-up patch; the old `sc[4096]` was an implicit context ceiling. |
| `engine/c/glm.c`, `stop_ids[8]` and `g_stop[9]` | configured EOS/stop list | Keep fixed, validate and cap the parsed list | The parser stores at most eight configured IDs and reserves one slot for the explicit EOS; follow-up fuzz coverage remains required. |
| `engine/c/glm.c`, `draft[64]`/`batch[64]` | speculative window | Keep fixed, clamp `g_draft` to 63 before use | The decode path clamps the user-controlled draft window; add a regression test when the C engine test target is available. |
| `engine/c/glm.c`, `ws[64]` and `missk[64]` | active experts per batch | Keep fixed, validate `topk`/working-set limits | These are routing work buffers, not tensor-dimension buffers; a complete adversarial-config test is still pending. |
| `engine/c/tok.h`, byte maps and UTF-8 scratch | byte alphabet and token encoding | Keep fixed by the byte-level tokenizer contract | The dimensions are protocol constants, independent of vocabulary size; tokenizer fuzz harness exists. |

## Verification boundary

The native C++ Release gate passed 251/251 tests after the previous follow-up.
This C-engine hardening patch cannot be compiled on the current Windows host
because `make`/MinGW is unavailable. It therefore does **not** claim the C
engine, ASan/UBSan, or fuzzing acceptance criteria are complete. The required
next receipt is `make check` plus sanitizer and fuzz runs on a host with the
engine toolchain.
