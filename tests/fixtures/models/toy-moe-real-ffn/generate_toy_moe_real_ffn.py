#!/usr/bin/env python3
"""Generates the toy-moe-real-ffn fixtures used to prove that the routed
expert's REAL FFN layer (gate_proj/up_proj/down_proj SwiGLU, issue #81.7c)
actually transforms the attention context before the output projection,
not just swaps the shared lm_head.weight (#81.7/#81.7b).

Layout mirrors tests/fixtures/models/toy-moe-real/ (issue #81.7/#88):
- toy-moe-real-ffn.safetensors: the base model. embedding.weight [4, 8] is
  one-hot on hidden dim 0 in EVERY row (not just the "alpha" row) -- the
  route-signature text DeepSeekMoEAdapter appends ("moe-route e0") hashes
  to an unpredictable vocabulary index via TokenIdFor, so every row must
  carry the same one-hot pattern for the oracle below to hold regardless
  of which row gets looked up (same trick #81.7b's kimi/minimax/glm
  fixtures use). The base's own lm_head.weight is a decoy that argmaxes to
  "alpha" (column 0) -- what the output would be if neither the expert's
  lm_head NOR its FFN were applied.
- expert0.safetensors (DeepSeek's default top-1 route for an empty
  prompt): real lm_head.weight PLUS real gate_proj/up_proj/down_proj.
  With a single-token prompt, attention over one key/value pair is
  trivially that pair's value -- the real, uniform one-hot embedding --
  so the FFN's input is deterministically known ahead of time: x = [1, 0,
  0, 0, 0, 0, 0, 0].

  gate_proj/up_proj are all-zero except column 0, row 0
  (gate_proj[0][0]=10.0, up_proj[0][0]=1.0), so:
    gate_out = gate_proj @ x  -> all zero except index 0 (= 10.0)
    up_out   = up_proj @ x    -> all zero except index 0 (= 1.0)
    hidden   = silu(gate_out) * up_out -> all zero except index 0
             (= silu(10.0) ~= 9.9995)
  down_proj is all-zero except column 0, row 3 (down_proj[3][0]=1.0), so:
    ffn_out = down_proj @ hidden -> all zero except index 3 (~= 9.9995)

  lm_head.weight's column 3 is [0.1, 0.2, 5.0, 0.3] (argmaxes to "gamma",
  index 2) -- since ffn_out is a POSITIVE scalar times a one-hot vector on
  dim 3, the sign/order of the dot product with each vocabulary row's
  column-3 value is preserved, so the final argmax is "gamma" regardless
  of the exact positive scale silu(10.0) evaluates to.
- expert1.safetensors: present for a genuine multi-shard manifest but
  unused by this fixture's default routing (DeepSeek's route logits pick
  expert 0 first for an empty/keyword-free prompt).

Run manually to regenerate:
    python3 tests/fixtures/models/toy-moe-real-ffn/generate_toy_moe_real_ffn.py
"""
import json
import os
import struct

VOCAB = ["alpha", "beta", "gamma", "delta"]
HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 16


def write_safetensors(path, tensors):
    """tensors: dict[name] -> (shape, flat_values)."""
    blobs = {}
    offset = 0
    header = {}
    for name, (shape, values) in tensors.items():
        blob = struct.pack(f"<{len(values)}f", *values)
        blobs[name] = blob
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)

    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for name in tensors:
            f.write(blobs[name])


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Base: embedding one-hot on dim 0 for every row; decoy lm_head
    # argmaxes to "alpha".
    embedding = []
    for _ in VOCAB:
        row = [0.0] * HIDDEN_SIZE
        row[0] = 1.0
        embedding.extend(row)

    decoy_lm_head = [0.0] * (len(VOCAB) * HIDDEN_SIZE)
    for token, value in enumerate([5.0, 0.1, 0.2, 0.3]):
        decoy_lm_head[token * HIDDEN_SIZE + 0] = value

    write_safetensors(
        os.path.join(out_dir, "toy-moe-real-ffn.safetensors"),
        {
            "embedding.weight": ((len(VOCAB), HIDDEN_SIZE), embedding),
            "lm_head.weight": ((len(VOCAB), HIDDEN_SIZE), decoy_lm_head),
        },
    )

    # Expert 0: real lm_head (column 3 argmaxes to "gamma") + real FFN.
    expert0_lm_head = [0.0] * (len(VOCAB) * HIDDEN_SIZE)
    for token, value in enumerate([0.1, 0.2, 5.0, 0.3]):
        expert0_lm_head[token * HIDDEN_SIZE + 3] = value

    gate_proj = [0.0] * (INTERMEDIATE_SIZE * HIDDEN_SIZE)
    gate_proj[0 * HIDDEN_SIZE + 0] = 10.0
    up_proj = [0.0] * (INTERMEDIATE_SIZE * HIDDEN_SIZE)
    up_proj[0 * HIDDEN_SIZE + 0] = 1.0
    down_proj = [0.0] * (HIDDEN_SIZE * INTERMEDIATE_SIZE)
    down_proj[3 * INTERMEDIATE_SIZE + 0] = 1.0

    write_safetensors(
        os.path.join(out_dir, "expert0.safetensors"),
        {
            "lm_head.weight": ((len(VOCAB), HIDDEN_SIZE), expert0_lm_head),
            "gate_proj.weight": ((INTERMEDIATE_SIZE, HIDDEN_SIZE), gate_proj),
            "up_proj.weight": ((INTERMEDIATE_SIZE, HIDDEN_SIZE), up_proj),
            "down_proj.weight": ((HIDDEN_SIZE, INTERMEDIATE_SIZE), down_proj),
        },
    )

    # Expert 1: present but unused by this fixture's default routing.
    expert1_lm_head = [0.0] * (len(VOCAB) * HIDDEN_SIZE)
    for token, value in enumerate([0.2, 0.1, 0.3, 5.0]):
        expert1_lm_head[token * HIDDEN_SIZE + 0] = value
    write_safetensors(
        os.path.join(out_dir, "expert1.safetensors"),
        {"lm_head.weight": ((len(VOCAB), HIDDEN_SIZE), expert1_lm_head)},
    )

    print("wrote toy-moe-real-ffn fixtures: base decoy argmaxes to 'alpha', "
          "expert0's real lm_head + FFN argmaxes to 'gamma'.")


if __name__ == "__main__":
    main()
