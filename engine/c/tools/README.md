# Tools

These scripts support model preparation and offline engineering work. They are
not runtime dependencies of the C engine.

- `convert_fp8_to_int4.py`, `download_glm52.py`: model preparation. The
  converter detects Kimi K2/its layer count from `config.json` and writes
  resumable shards atomically.
- `make_glm_oracle.py`, `make_kimi_oracle.py`, `make_glm_bench_model.py`:
  deterministic fixtures
- `benchmark_cuda_fixture.py`, `eval_glm.py`, `fetch_benchmarks.py`: benchmarks
- `gen_unicode.py`: tokenizer table generation

Run them from `c/`, for example:

```sh
python3 tools/convert_fp8_to_int4.py --selftest
python3 tools/make_kimi_oracle.py
python3 tools/make_glm_bench_model.py --output /tmp/colibri-bench
```
