#!/bin/sh
set -eu

run_corpus() {
    target=$1
    corpus=$2
    find "$corpus" -type f -print | LC_ALL=C sort | while IFS= read -r seed; do
        "$target" "$seed"
    done
}

run_corpus fuzz/bin/fuzz_json_replay fuzz/corpus/json
run_corpus fuzz/bin/fuzz_safetensors_replay fuzz/corpus/safetensors
run_corpus fuzz/bin/fuzz_tokenizer_replay fuzz/corpus/tokenizer
run_corpus fuzz/bin/fuzz_serve_protocol_replay fuzz/corpus/serve_protocol

echo "engine fuzz smoke: all seed replays passed"
