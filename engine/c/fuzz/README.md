# Engine C fuzzing

Issue #122 adds four harnesses for the vendored C engine's untrusted-input
boundaries:

- `fuzz_json.c`: length-bounded `json.h` parsing;
- `fuzz_safetensors.c`: safetensors header and tensor-index metadata;
- `fuzz_tokenizer.c`: arbitrary-byte encode/decode against a valid tokenizer;
- `fuzz_serve_protocol.c`: the engine's length-prefixed line protocol.

The committed corpus contains valid and malformed inputs for each grammar.
Run the deterministic Linux sanitizer replay with:

```sh
make -C engine/c fuzz-smoke
```

The smoke target builds the same `LLVMFuzzerTestOneInput` entry points with
GCC ASan+UBSan, replays every seed, and applies 64 deterministic
bit-flip/truncation/extension mutations per seed. It is intentionally budgeted
and is not evidence of a long coverage-guided campaign.

When Clang with libFuzzer is available:

```sh
make -C engine/c fuzz-libfuzzer
engine/c/fuzz/bin/fuzz_json \
  -max_total_time=3600 engine/c/fuzz/corpus/json
```

Repeat the command for `fuzz_safetensors`, `fuzz_tokenizer`, and
`fuzz_serve_protocol` with the corresponding corpus. Record libFuzzer's final
statistics externally; no one-hour statistics are committed or claimed by the
bounded smoke target.

`fuzz_tokenizer` resolves its tokenizer from `COLI_FUZZ_TOKENIZER`, defaulting
to the repository's minimal engine fixture when run from `engine/c`.
