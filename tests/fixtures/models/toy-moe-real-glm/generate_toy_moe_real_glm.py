#!/usr/bin/env python3
"""Generates the toy-moe-real-glm fixtures used to prove that GlmMoEAdapter's
routed expert actually changes the forward's output (issue #81.7b), the same
way #81.7/#88 already proved it for DeepSeekMoEAdapter.

- toy-moe-real-glm.safetensors: the base model, embedding.weight [4, 8]
  (every row one-hot on hidden dim 0, so the result is independent of which
  vocabulary row the prompt's route-signature text happens to hash to) and
  a DECOY lm_head.weight [4, 8] whose column 0 argmaxes to "alpha" (index
  0).
- expert0/expert1/expert2.safetensors: real per-expert shards
  (moe_expert_shards in model.us4manifest). GlmMoEAdapter's default route
  logits ({0.65, 0.55, 0.80, 0.45}, no keyword match for an empty prompt)
  make the router pick expert INDEX 2 by default -- so expert2's
  lm_head.weight column 0 argmaxes to "beta" (index 1), deliberately
  different from the base decoy, while expert0/expert1 are present but
  unused by this default routing.

Run manually to regenerate:
    python3 tests/fixtures/models/toy-moe-real-glm/generate_toy_moe_real_glm.py
"""
import json
import os
import struct

VOCAB = ["alpha", "beta", "gamma", "delta"]
HIDDEN_SIZE = 8


def write_safetensors(path, embedding_col0, lm_head_col0):
    embedding = [0.0] * (len(VOCAB) * HIDDEN_SIZE)
    if embedding_col0 is not None:
        for token, value in enumerate(embedding_col0):
            embedding[token * HIDDEN_SIZE + 0] = value

    lm_head = [0.0] * (len(VOCAB) * HIDDEN_SIZE)
    for token, value in enumerate(lm_head_col0):
        lm_head[token * HIDDEN_SIZE + 0] = value

    embedding_bytes = struct.pack(f"<{len(embedding)}f", *embedding)
    lm_head_bytes = struct.pack(f"<{len(lm_head)}f", *lm_head)

    embedding_bytes_written = embedding_bytes if embedding_col0 is not None else b""
    lm_head_start = len(embedding_bytes_written)

    header = {
        "lm_head.weight": {
            "dtype": "F32",
            "shape": [len(VOCAB), HIDDEN_SIZE],
            "data_offsets": [lm_head_start, lm_head_start + len(lm_head_bytes)],
        },
    }
    if embedding_col0 is not None:
        header["embedding.weight"] = {
            "dtype": "F32",
            "shape": [len(VOCAB), HIDDEN_SIZE],
            "data_offsets": [0, len(embedding_bytes)],
        }
    header_json = json.dumps(header).encode("utf-8")

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(embedding_bytes_written)
        f.write(lm_head_bytes)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    write_safetensors(
        os.path.join(out_dir, "toy-moe-real-glm.safetensors"),
        embedding_col0=[1.0, 1.0, 1.0, 1.0],
        lm_head_col0=[5.0, 0.1, 0.2, 0.3],
    )
    write_safetensors(
        os.path.join(out_dir, "expert0.safetensors"),
        embedding_col0=None,
        lm_head_col0=[0.1, 0.2, 5.0, 0.3],
    )
    write_safetensors(
        os.path.join(out_dir, "expert1.safetensors"),
        embedding_col0=None,
        lm_head_col0=[0.1, 0.2, 0.3, 5.0],
    )
    write_safetensors(
        os.path.join(out_dir, "expert2.safetensors"),
        embedding_col0=None,
        lm_head_col0=[0.1, 5.0, 0.2, 0.3],
    )

    print("wrote toy-moe-real-glm fixtures: base decoy argmaxes to 'alpha', "
          "expert2 (the router's default pick) argmaxes to 'beta'.")


if __name__ == "__main__":
    main()
