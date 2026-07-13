#!/usr/bin/env python3
"""Generates the toy-dense-real-draft.safetensors fixture used to prove
speculative decoding runs a genuine forward over a real (smaller) draft
model, instead of fabricating a draft proposal by copying the target's
output and incrementing the last token.

Shares the same 4-token vocabulary as toy-dense-real.safetensors (alpha,
beta, gamma, delta) but uses a smaller hidden size (2 instead of 8), as a
real draft model would. Only embedding row 3 ("delta") and lm_head's
hidden dim 0 carry non-zero values: embedding("delta") is one-hot on
hidden dim 0, and lm_head[:, 0] = [0.1, 0.2, 0.3, 5.0], so the draft
model's own real forward argmaxes to "delta" (index 3) when the previous
token is "delta" -- deliberately matching what the target model actually
generates for prompt "alpha" (see generate_toy_dense_real.py), so the
speculative decoder's ACCEPT path is exercised by genuine computation
rather than a coincidence of a scripted mock.

Run manually to regenerate:
    python3 tests/fixtures/models/toy-dense-real/generate_toy_draft_model.py
"""
import json
import os
import struct

VOCAB = ["alpha", "beta", "gamma", "delta"]
DRAFT_HIDDEN_SIZE = 2


def main():
    embedding = [0.0] * (len(VOCAB) * DRAFT_HIDDEN_SIZE)
    embedding[3 * DRAFT_HIDDEN_SIZE + 0] = 1.0  # "delta" one-hot on hidden 0

    logit_per_token = [0.1, 0.2, 0.3, 5.0]  # argmax -> "delta" (index 3)
    lm_head = [0.0] * (len(VOCAB) * DRAFT_HIDDEN_SIZE)
    for token in range(len(VOCAB)):
        lm_head[token * DRAFT_HIDDEN_SIZE + 0] = logit_per_token[token]

    embedding_bytes = struct.pack(f"<{len(embedding)}f", *embedding)
    lm_head_bytes = struct.pack(f"<{len(lm_head)}f", *lm_head)

    header = {
        "embedding.weight": {
            "dtype": "F32",
            "shape": [len(VOCAB), DRAFT_HIDDEN_SIZE],
            "data_offsets": [0, len(embedding_bytes)],
        },
        "lm_head.weight": {
            "dtype": "F32",
            "shape": [len(VOCAB), DRAFT_HIDDEN_SIZE],
            "data_offsets": [len(embedding_bytes),
                             len(embedding_bytes) + len(lm_head_bytes)],
        },
    }
    header_json = json.dumps(header).encode("utf-8")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "toy-dense-real-draft.safetensors"),
              "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(embedding_bytes)
        f.write(lm_head_bytes)

    print("wrote toy-dense-real-draft.safetensors: draft forward for "
          "previous token 'delta' argmaxes to 'delta' -- matches the "
          "target model's real output, so the speculative ACCEPT path is "
          "exercised by genuine computation.")


if __name__ == "__main__":
    main()
