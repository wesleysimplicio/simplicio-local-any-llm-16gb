# Colibri correctness fixtures

`run_engine_forward_oracles.py` is the dependency-free local correctness
suite for the vendored C engine. It builds `engine/c/glm`, then checks:

- teacher-forcing 32/32 and greedy generation 20/20;
- absorption enabled/disabled parity;
- DSA forced to retain every key versus dense attention;
- n-gram speculative decoding versus non-speculative generation.

GLM uses seed `1234` from `engine/c/tools/make_glm_oracle.py`; DeepSeek uses
seed `4202` from `engine/c/tools/make_deepseek_oracle.py`. Their generated
tiny checkpoints and JSON references are committed under `colibri/`, so
the automated suite does not import Torch, Transformers, safetensors, or
access the network.

Regenerate a fixture only after an intentional forward-contract change:

1. Run the corresponding generator from `engine/c/` in a disposable
   environment containing its offline generation dependencies.
2. Copy the generated checkpoint and `ref_*.json` into this directory.
3. Run `python3 tests/fixtures/engine/run_engine_forward_oracles.py`.
4. Review the binary size and generated token diff before committing.

Kimi is declared by the runner but skipped with an explicit
`SKIP checkpoint kimi_tiny` message until `colibri/kimi_tiny/` and
`colibri/ref_kimi.json` are available from the Kimi-family implementation.
Missing compilers, Make, broken fixtures, or oracle mismatches are failures,
never skips.

`engine/c/tests/test_quant.c` validates committed int8/int4/int2 packed-byte
vectors and compares the production scalar and IDOT kernels. This keeps
quantization coverage in `make test` without Python package dependencies.
