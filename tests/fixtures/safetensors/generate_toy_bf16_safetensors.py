#!/usr/bin/env python3
"""Generates a genuine minimal .safetensors fixture with a real BF16 tensor
for safetensors_reader_contract_test.cpp (issue #81.11: production
checkpoints -- e.g. Qwen2.5-0.5B's model.safetensors -- store weights as
BF16, which SafetensorsReader::ReadFloat32 rejected before this fixture's
test was added).

Values are restricted to exact powers of two (1.0, -1.0, 2.0, -2.0, 0.5,
-0.5, 4.0, -4.0) so bf16 truncation (keep the top 16 bits of the float32
bit pattern, zero the rest) is lossless for them -- the oracle in the test
can assert exact equality instead of a tolerance.

Run manually to regenerate:
    python3 tests/fixtures/safetensors/generate_toy_bf16_safetensors.py
"""
import json
import os
import struct

SHAPE = (2, 4)
VALUES = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 4.0, -4.0]


def to_bf16_bytes(values):
    out = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bf16 = (bits >> 16) & 0xFFFF
        out += struct.pack("<H", bf16)
    return bytes(out)


def main():
    tensor_bytes = to_bf16_bytes(VALUES)

    header = {
        "embedding.weight": {
            "dtype": "BF16",
            "shape": list(SHAPE),
            "data_offsets": [0, len(tensor_bytes)],
        },
    }
    header_json = json.dumps(header).encode("utf-8")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "toy_bf16.safetensors")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(tensor_bytes)

    print(f"wrote {out_path}: {len(header_json)}-byte header, "
          f"{len(tensor_bytes)} bytes of BF16 tensor data")


if __name__ == "__main__":
    main()
