#!/usr/bin/env python3
"""Generates the toy-dense-real.safetensors fixture used by the
"run uses real weights end to end" Playwright test.

Writes a real safetensors file with an embedding.weight [4, 8] and a
lm_head.weight [4, 8] tensor matching the runtime's fixed kHiddenSize=8
dense path (runtime/adapters/dense_adapter_base.cpp). lm_head.weight uses
the real HuggingFace/safetensors convention for nn.Linear(hidden, vocab)
-- [vocab_size, hidden_size], i.e. [out_features, in_features] -- which
TryRealOutputProjection transposes internally; this is NOT the runtime's
internal [hiddenSize, vocab] matmul buffer layout. Only the "alpha" row
(vocab index 0) and hidden dim 0 of every lm_head row carry non-zero
values, chosen so the expected next token is unambiguous: embedding("alpha")
is one-hot on hidden dim 0, so logits[token] = lm_head[token, 0] =
[0.1, 0.2, 0.3, 5.0], and argmax picks "delta" (index 3) with a wide margin.

Run manually to regenerate:
    python3 tests/fixtures/models/toy-dense-real/generate_toy_dense_real.py
"""
import json
import os
import struct

VOCAB = ["alpha", "beta", "gamma", "delta"]
HIDDEN_SIZE = 8


def main():
    embedding = [0.0] * (len(VOCAB) * HIDDEN_SIZE)
    embedding[0 * HIDDEN_SIZE + 0] = 1.0  # "alpha" is one-hot on hidden dim 0

    logit_per_token = [0.1, 0.2, 0.3, 5.0]
    lm_head = [0.0] * (len(VOCAB) * HIDDEN_SIZE)
    for token in range(len(VOCAB)):
        lm_head[token * HIDDEN_SIZE + 0] = logit_per_token[token]

    embedding_bytes = struct.pack(f"<{len(embedding)}f", *embedding)
    lm_head_bytes = struct.pack(f"<{len(lm_head)}f", *lm_head)

    header = {
        "embedding.weight": {
            "dtype": "F32",
            "shape": [len(VOCAB), HIDDEN_SIZE],
            "data_offsets": [0, len(embedding_bytes)],
        },
        "lm_head.weight": {
            "dtype": "F32",
            "shape": [len(VOCAB), HIDDEN_SIZE],
            "data_offsets": [len(embedding_bytes),
                             len(embedding_bytes) + len(lm_head_bytes)],
        },
    }
    header_json = json.dumps(header).encode("utf-8")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "toy-dense-real.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(embedding_bytes)
        f.write(lm_head_bytes)

    print("wrote toy-dense-real.safetensors: expected argmax next token for "
          "'alpha' is 'delta' (index 3), logits = [0.1, 0.2, 0.3, 5.0]")


if __name__ == "__main__":
    main()
