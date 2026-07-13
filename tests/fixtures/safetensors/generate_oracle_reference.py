#!/usr/bin/env python3
"""Generates the independent-oracle reference logits for
oracle_correctness_contract_test.cpp.

Loads the REAL tensors from toy_real.safetensors (the same genuine binary
fixture used by safetensors_reader_contract_test.cpp) with a from-scratch
safetensors parser written independently of the C++ SafetensorsReader, then
computes logits = embedding[token_id] @ lm_head for a fixed set of token
ids using plain Python arithmetic (no numpy dependency, so this stays
runnable with only the standard library). The output is the external
oracle the C++ forward path (embedding lookup + ScalarMatmul) must match
token-exact within a documented tolerance.

Run manually to regenerate:
    python3 tests/fixtures/safetensors/generate_oracle_reference.py
"""
import json
import os
import struct


def load_safetensors(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            count = (end - begin) // 4
            values = struct.unpack(f"<{count}f", raw)
            tensors[name] = {"shape": info["shape"], "values": list(values)}
    return tensors


def matmul_row(row, matrix_values, matrix_shape):
    rows, cols = matrix_shape
    assert len(row) == rows
    result = [0.0] * cols
    for r in range(rows):
        for c in range(cols):
            result[c] += row[r] * matrix_values[r * cols + c]
    return result


def main():
    fixture_dir = os.path.dirname(os.path.abspath(__file__))
    tensors = load_safetensors(os.path.join(fixture_dir, "toy_real.safetensors"))

    embedding = tensors["embedding.weight"]
    lm_head = tensors["lm_head.weight"]
    vocab_size, hidden_size = embedding["shape"]

    reference = {}
    for token_id in range(vocab_size):
        row = embedding["values"][token_id * hidden_size:(token_id + 1) * hidden_size]
        logits = matmul_row(row, lm_head["values"], lm_head["shape"])
        reference[str(token_id)] = logits

    out_path = os.path.join(fixture_dir, "oracle_reference_logits.json")
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {len(reference)} reference logit rows to {out_path}")


if __name__ == "__main__":
    main()
