#!/usr/bin/env python3
"""Issue #81.11: reference-generation script for validating this runtime's
SafetensorsReader against a REAL, production-scale checkpoint, instead of
only the tiny hand-built fixtures the rest of this session's issues used.

This script does NOT depend on torch/transformers -- it parses the
safetensors binary format directly (same approach as
runtime/core/safetensors_reader.cpp) so the "oracle" it produces is
genuinely independent of the C++ reader it's meant to check, not a
reimplementation that could share the same bug.

Usage (the checkpoint itself is intentionally NOT committed to this repo --
988MB of BF16 weights has no place in git history; download it yourself):

    pip install huggingface_hub
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen2.5-0.5B', 'model.safetensors', local_dir='/tmp/qwen2.5-0.5b')
"
    python3 tests/fixtures/real_checkpoint/generate_reference_from_real_checkpoint.py \
        /tmp/qwen2.5-0.5b/model.safetensors

Then compile and run the C++ side against the same file (see
tests/fixtures/real_checkpoint/README.md for the exact commands and the
findings from doing this once, captured there so this isn't a "trust me"
claim) and diff the printed rows against this script's output.
"""
import argparse
import array
import json
import struct
import sys


def bf16_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("safetensors_path")
    parser.add_argument("--token-ids", type=int, nargs="*", default=[0, 9707])
    args = parser.parse_args()

    with open(args.safetensors_path, "rb") as f:
        header_length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_length))
        data_start = 8 + header_length

        print(f"tensor_count={len([k for k in header if k != '__metadata__'])}")

        embedding_key = "model.embed_tokens.weight"
        if embedding_key not in header:
            print(f"'{embedding_key}' not found -- inspect the header keys "
                  "manually; naming varies by architecture", file=sys.stderr)
            sys.exit(1)

        info = header[embedding_key]
        print(f"dtype={info['dtype']} shape={info['shape']}")
        vocab_size, hidden_size = info["shape"]
        begin, _end = info["data_offsets"]
        row_bytes = hidden_size * 2  # BF16 = 2 bytes/element

        for token_id in args.token_ids:
            if token_id >= vocab_size:
                continue
            f.seek(data_start + begin + token_id * row_bytes)
            raw = f.read(row_bytes)
            values = array.array("H")
            values.frombytes(raw)
            row = [bf16_to_f32(v) for v in values]
            print(f"token{token_id} first8: {row[:8]}")
            print(f"token{token_id} last8: {row[-8:]}")


if __name__ == "__main__":
    main()
