#!/usr/bin/env python3
"""Generate deterministic tokenizer parity vectors from an offline HF snapshot."""

import argparse
import hashlib
import json
from pathlib import Path


BASE_CASES = [
    ("empty", ""),
    ("ascii", "hello world"),
    ("uppercase", "HELLO World"),
    ("contractions", "I'm sure we'll test what isn't obvious."),
    ("spaces_2", "a  b"),
    ("spaces_8", "a        b"),
    ("leading_spaces", "   leading"),
    ("trailing_spaces", "trailing   "),
    ("tabs", "a\tb\tc"),
    ("newlines", "line 1\nline 2\r\nline 3"),
    ("digits_short", "0 12 345"),
    ("digits_long", "123456789012345678901234567890"),
    ("punctuation", "!@#$%^&*()[]{};:,.?"),
    ("python", "def café(x):\n    return x ** 2\n"),
    ("c_code", "int main(void) { return 0; }"),
    ("json", '{"olá":true,"n":123,"emoji":"🙂"}'),
    ("markdown", "## Title\n- **bold** and `code`"),
    ("url", "https://example.com/a?q=olá&x=1"),
    ("email", "user+tag@example.com"),
    ("pt_br", "ação, coração, órgão, bênção, você"),
    ("combining", "Cafe\u0301 nai\u0308ve"),
    ("precomposed", "Café naïve"),
    ("emoji", "🙂🚀🧠"),
    ("emoji_zwj", "👩‍💻👨‍👩‍👧‍👦"),
    ("flags", "🇧🇷🇯🇵"),
    ("cjk", "你好，世界。日本語"),
    ("arabic", "مرحبا بالعالم"),
    ("devanagari", "नमस्ते दुनिया"),
    ("cyrillic", "Привет, мир"),
    ("greek", "Γειά σου Κόσμε"),
    ("math", "∀x∈ℝ: x² ≥ 0; ∑∞"),
    ("currency", "R$ 1.234,56 € £ ¥"),
    ("nbsp", "a\u00a0b"),
    ("thin_space", "a\u2009b"),
    ("line_separator", "a\u2028b"),
    ("null_word", "NUL is written as text, not a byte"),
    ("slashes", r"c:\temp\\file /tmp/file"),
    ("quotes", "\"double\" 'single' `tick`"),
    ("mixed", "Olá, 世界! 🙂 123\nnext"),
    ("repeat", "abababababababababab"),
]


def adversarial_cases(special_token):
    return BASE_CASES + [
        ("special_only", special_token),
        ("special_middle", f"before{special_token}after"),
        ("special_repeated", f"{special_token}{special_token}"),
        ("special_unicode", f"ação🙂{special_token}世界"),
    ]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("glm", "deepseek", "kimi"))
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--special-token", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "transformers is required only to regenerate real HF vectors; "
            "install it outside the runtime environment") from error

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False)
    vectors = []
    for name, text in adversarial_cases(args.special_token):
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        vectors.append({"name": name, "text": text, "ids": ids,
                        "reference_decode": decoded})
    tokenizer_json = args.model_dir / "tokenizer.json"
    config_json = args.model_dir / "tokenizer_config.json"
    document = {
        "schema_version": 1,
        "family": args.family,
        "source": "huggingface-offline-snapshot",
        "tokenizer_sha256": sha256(tokenizer_json),
        "tokenizer_config_sha256": sha256(config_json) if config_json.is_file() else None,
        "cases": vectors,
    }
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
