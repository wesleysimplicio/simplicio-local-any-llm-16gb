# Tokenizer parity fixtures

`tok_vectors_{glm,deepseek,kimi}.json` contains 44 deterministic adversarial
cases per family. The committed `contract_tokenizer_*.json` files are small,
dependency-free byte-BPE tokenizers used to exercise the C vector runner,
special-token handling, Unicode, and encode/decode round trips in `make test`.
They are explicitly marked `synthetic-byte-bpe-contract`; they are not evidence
of parity with a released checkpoint.

Real Hugging Face parity vectors are generated once from an already downloaded
snapshot, without network access:

```bash
python tools/make_tokenizer_vectors.py \
  --family glm \
  --model-dir /path/to/glm-snapshot \
  --special-token '<|user|>' \
  --output tests/fixtures/tokenizer/tok_vectors_glm.json
```

Repeat for DeepSeek and Kimi with each model's actual special token and run
`make test-tokenizer`. The generator records SHA-256 hashes of
`tokenizer.json` and `tokenizer_config.json` so the reference is auditable.
