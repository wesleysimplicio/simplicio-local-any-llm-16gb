# Real checkpoint validation (issue #81.11 / #106)

Every fixture used by #82-#91/#102-#105/#108 is a tiny, hand-built
`.safetensors` file (vocab of 4 tokens, `hiddenSize` of 4 or 8). That's real
math over real data and correct for what it tests, but none of it was ever
checked against a real, production-scale checkpoint downloaded from
Hugging Face. This directory documents doing that once, honestly, and what
it actually found.

## What was validated

Checkpoint: [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B)
(Apache-2.0), `model.safetensors`, 988MB, 290 tensors, `hidden_size=896`,
`vocab_size=151936`, BF16 weights, tied embeddings (no separate
`lm_head.weight`). Downloaded via `huggingface_hub`, **not committed to
this repo** -- 988MB of weights has no place in git history, and the DoD
for #106 explicitly calls for checking license/size before ever
considering that. `generate_reference_from_real_checkpoint.py` in this
directory is the versioned, runnable part: an independent Python oracle
that parses the safetensors binary format directly (no torch/transformers,
no reuse of this runtime's C++ parsing code) and prints the real
`model.embed_tokens.weight` row for a given token id.

### 1. SafetensorsReader parses the real, production-scale header and reads real bytes correctly

Ran `SafetensorsReader::Open` + `ReadFloat32("model.embed_tokens.weight")`
against the actual downloaded file (a throwaway C++ program linked against
`libus4_runtime_core.a`, not checked in -- the point was to prove the
*existing* reader handles this, not to add a permanent binary artifact).
Result:

```
tensor_count=290
dtype=BF16 shape=[151936,896]
read_error= values_size=136134656
token0 first8: -0.0100708 0.0419922 0.0098877 0.000904083 -0.0275879 -0.00234985 -0.00139618 -0.0198975
token9707 first8: -0.0251465 0.0055542 -0.0101929 0.0128174 -4.33922e-05 -0.00497437 -0.0127563 0.00491333
```

This matches `generate_reference_from_real_checkpoint.py`'s independent
oracle byte-for-byte (both scripts derive from the same raw bytes, computed
two different ways). Before this validation, `SafetensorsReader::ReadFloat32`
only accepted `dtype == "F32"` and rejected `"BF16"` outright -- real
checkpoints (this one included) store weights as BF16. Fixed as a real
code change (`Bf16ToFloat32`, exact widening: bf16 is the top 16 bits of a
float32, so shifting left 16 and reinterpreting is lossless for the bits
kept), with a fixture-and-oracle unit test
(`SafetensorsReaderContractTest.ReadsRealBf16TensorBytesAsFloat32`) that
doesn't depend on the downloaded checkpoint being present.

### 2. LoadModelAsset + the qwen adapter correctly, safely fall back on this real file -- and that fallback exposes real scale limitations

```
load_ok=1 error=
has_real_weights=0
real_tensors_count=0
load_status=real-header-parsed-no-known-tensor-names
used_real_weights=0
text=us4 apple
```

`LoadModelAsset` opens the real file and parses its header successfully,
but finds none of the two hardcoded tensor names it looks for
(`"embedding.weight"`, `"lm_head.weight"`) -- Qwen2.5's real tensor is
named `model.embed_tokens.weight`, and because `tie_word_embeddings=true`
there is no separate output-projection tensor at all. `hasRealWeights`
stays `false`, and `Generate()` correctly, visibly falls back to the fully
synthetic path (`used_real_weights=false`) instead of crashing or silently
mixing real and synthetic data. This is the *correct* behavior given the
mismatch -- but the mismatch itself is real and worth naming precisely.

## Identified scale limitations (documented as technical debt, per #106's DoD)

These are concrete, not speculative -- each one was hit while trying to run
this runtime against the real checkpoint above:

1. **Fixed tensor names.** `LoadModelAsset` only ever looks for
   `"embedding.weight"`/`"lm_head.weight"`. Real HF checkpoints use
   architecture-specific names (`model.embed_tokens.weight`, `lm_head.weight`
   only when embeddings aren't tied, etc.). A real loader would need a
   per-family name mapping (or a `tie_word_embeddings` flag reusing the
   embedding tensor for the projection).
2. **BF16 was unsupported until this issue.** Fixed here; F32/BF16 are now
   both readable. Other real dtypes (FP16, FP8 variants, quantized formats)
   remain unsupported and would need the same treatment if a real
   checkpoint uses them.
3. **Fixed scaffold `hiddenSize` (8) in `DenseAdapterBase::Generate`.**
   `kHiddenSize` is a compile-time constant, not derived from the asset.
   Qwen2.5-0.5B's real `hidden_size=896` can never match it, so
   `TryRealEmbeddingRow`/`TryRealOutputProjection`'s shape checks always
   fail for a real checkpoint of this family, regardless of tensor naming.
   Making hidden size asset-driven throughout `dense_adapter_base.cpp`
   (buffers, tensor shapes, the per-step argmax loop) is a real refactor,
   not a one-line fix -- out of scope here, tracked as debt.
4. **In-memory string vocabulary can't represent a real ~150k-token
   vocab.** `ModelAsset::vocabulary` is a `std::vector<std::string>`
   populated from a manifest's CSV `vocabulary=` field or an adapter's
   hardcoded word list. Real tokenizer vocabularies (151936 entries for
   Qwen2.5) would need to come from the real `tokenizer.json`/BPE merges
   (already parseable via `BpeTokenizer`, see #84) driving *token ids*
   directly, not a CSV list of literal words matched by substring.
5. **No real multi-layer transformer stack.** Even with 1-3 fixed, the
   forward this runtime runs is a single embedding lookup -> one
   attention step -> one output projection -- not the real model's 24
   transformer blocks (attention + RMSNorm + SwiGLU FFN per layer, see
   #104 for the one-expert-FFN-layer version of this gap in the MoE path).
   Matching the real model's actual generation quality would require
   implementing that full stack, which is a project-scale undertaking, not
   something to retrofit onto this session.

None of these are fixed here -- per #106's own DoD, identifying and
documenting them (rather than pretending they don't exist, or spending the
rest of the session on a full transformer reimplementation) is the honest
outcome. #86 already established the epic's precedent for this: report a
real, hardware/scope blocker plainly instead of fabricating success.
