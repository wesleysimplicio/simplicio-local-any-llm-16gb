#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${TMPDIR:-/tmp}/us4-adaptive-cache-speculation-tiny"

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  -I"${repo_root}/runtime" \
  "${repo_root}/tests/tiny/adaptive_cache_speculation_tiny.cpp" \
  "${repo_root}/runtime/moe/expert_pager.cpp" \
  "${repo_root}/runtime/speculative/peagle_decoder.cpp" \
  "${repo_root}/runtime/speculative/lossless_speculative_session.cpp" \
  "${repo_root}/runtime/speculative/speculative_telemetry.cpp" \
  -o "${binary}"

"${binary}"
