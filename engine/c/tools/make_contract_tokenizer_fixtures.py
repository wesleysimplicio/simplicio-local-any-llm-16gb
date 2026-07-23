#!/usr/bin/env python3
"""Regenerate dependency-free byte-BPE contract fixtures used by make test."""

import json
from pathlib import Path

from make_tokenizer_vectors import adversarial_cases


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "tests" / "fixtures" / "tokenizer"
SPECIALS = {
    "glm": [
        "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>",
        "<think>", "</think>", "<|endoftext|>",
    ],
    "deepseek": [
        "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>",
        "<｜end▁of▁sentence｜>", "<think>", "</think>",
    ],
    "kimi": [
        "<|im_system|>", "<|im_user|>", "<|im_assistant|>",
        "<|im_middle|>", "<|im_end|>",
    ],
}
CASE_SPECIAL = {
    "glm": "<|user|>",
    "deepseek": "<｜User｜>",
    "kimi": "<|im_user|>",
}


def byte_to_unicode():
    direct = list(range(ord("!"), ord("~") + 1))
    direct += list(range(ord("¡"), ord("¬") + 1))
    direct += list(range(ord("®"), ord("ÿ") + 1))
    values = list(direct)
    codepoints = list(direct)
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(values, (chr(codepoint) for codepoint in codepoints)))


def encode_contract(text, special_ids):
    ids = []
    cursor = 0
    ordered = sorted(special_ids, key=len, reverse=True)
    while cursor < len(text):
        match = next((token for token in ordered if text.startswith(token, cursor)), None)
        if match:
            ids.append(special_ids[match])
            cursor += len(match)
            continue
        next_special = min(
            (position for token in ordered
             if (position := text.find(token, cursor)) >= 0),
            default=len(text),
        )
        ids.extend(text[cursor:next_special].encode("utf-8"))
        cursor = next_special
    return ids


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    byte_map = byte_to_unicode()
    vocab = {byte_map[byte]: byte for byte in range(256)}
    for family, specials in SPECIALS.items():
        special_ids = {token: 256 + index for index, token in enumerate(specials)}
        tokenizer = {
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [
                {"id": token_id, "content": token}
                for token, token_id in special_ids.items()
            ],
        }
        vectors = {
            "schema_version": 1,
            "family": family,
            "source": "synthetic-byte-bpe-contract",
            "limitations": "Replace with make_tokenizer_vectors.py output for real HF parity.",
            "cases": [
                {"name": name, "text": text, "ids": encode_contract(text, special_ids)}
                for name, text in adversarial_cases(CASE_SPECIAL[family])
            ],
        }
        (OUTPUT / f"contract_tokenizer_{family}.json").write_text(
            json.dumps(tokenizer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (OUTPUT / f"tok_vectors_{family}.json").write_text(
            json.dumps(vectors, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
