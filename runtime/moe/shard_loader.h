#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace us4 {

// Shard-aware MoE loader contract.
// Expert weights for production models are routinely sharded across many
// files. The runtime needs a deterministic way to enumerate, identify, and
// route each shard before loading anything heavy. Sprint 08 introduces this
// contract surface; production reading happens in later sprints.

struct ExpertShardDescriptor {
  std::string family;       // "deepseek", "kimi", "minimax", etc.
  std::string modelId;
  std::size_t expertIndex = 0;
  std::size_t shardIndex = 0;
  std::size_t shardCount = 0;
  std::string filePath;
  std::string weightFormat; // "safetensors", "gguf"
  bool routedOnly = false;  // shared experts have routedOnly = false.
};

// Parse a shard manifest path like
// `tests/fixtures/models/deepseek-v3/shard-manifest.us4manifest` and produce
// a deterministic list of expert shards. Returns std::nullopt when the
// manifest is missing required fields.
std::optional<std::vector<ExpertShardDescriptor>>
ParseExpertShardManifest(const std::string &manifestPath);

// In-memory variant for tests; parses a raw manifest body without touching
// the filesystem. The contract mirrors the file-based parser so unit tests
// can keep the runtime contract honest.
std::optional<std::vector<ExpertShardDescriptor>>
ParseExpertShardManifestBody(const std::string &manifestBody);

} // namespace us4
