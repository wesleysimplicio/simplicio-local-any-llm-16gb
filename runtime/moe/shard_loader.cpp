#include "moe/shard_loader.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace us4 {

namespace {

std::string Trim(std::string_view value) {
  std::size_t begin = 0;
  std::size_t end = value.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return std::string(value.substr(begin, end - begin));
}

std::size_t ParseSize(const std::string &value) {
  if (value.empty()) {
    return 0;
  }
  return static_cast<std::size_t>(std::strtoul(value.c_str(), nullptr, 10));
}

bool ParseBool(const std::string &value) {
  if (value == "true" || value == "1" || value == "yes") {
    return true;
  }
  return false;
}

} // namespace

std::optional<std::vector<ExpertShardDescriptor>>
ParseExpertShardManifestBody(const std::string &manifestBody) {
  std::vector<ExpertShardDescriptor> shards;
  std::istringstream stream(manifestBody);
  std::string line;

  ExpertShardDescriptor current;
  bool insideShard = false;
  std::string family;
  std::string modelId;

  while (std::getline(stream, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }
    const auto eq = trimmed.find('=');
    if (eq == std::string::npos) {
      if (trimmed == "[shard]") {
        if (insideShard) {
          if (current.expertIndex == 0 && current.shardIndex == 0 &&
              current.filePath.empty()) {
            return std::nullopt;
          }
          shards.push_back(current);
        }
        current = ExpertShardDescriptor{};
        current.family = family;
        current.modelId = modelId;
        insideShard = true;
      }
      continue;
    }
    const std::string key = Trim(trimmed.substr(0, eq));
    const std::string value = Trim(trimmed.substr(eq + 1));
    if (!insideShard) {
      if (key == "family") {
        family = value;
      } else if (key == "model_id") {
        modelId = value;
      }
      continue;
    }
    if (key == "expert_index") {
      current.expertIndex = ParseSize(value);
    } else if (key == "shard_index") {
      current.shardIndex = ParseSize(value);
    } else if (key == "shard_count") {
      current.shardCount = ParseSize(value);
    } else if (key == "file") {
      current.filePath = value;
    } else if (key == "weight_format") {
      current.weightFormat = value;
    } else if (key == "routed_only") {
      current.routedOnly = ParseBool(value);
    } else if (key == "family") {
      current.family = value;
    } else if (key == "model_id") {
      current.modelId = value;
    }
  }
  if (insideShard) {
    shards.push_back(current);
  }
  if (shards.empty()) {
    return std::nullopt;
  }
  for (const auto &shard : shards) {
    if (shard.family.empty() || shard.modelId.empty() ||
        shard.filePath.empty() || shard.weightFormat.empty() ||
        shard.shardCount == 0) {
      return std::nullopt;
    }
  }
  return shards;
}

std::optional<std::vector<ExpertShardDescriptor>>
ParseExpertShardManifest(const std::string &manifestPath) {
  std::ifstream stream(manifestPath);
  if (!stream.is_open()) {
    return std::nullopt;
  }
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return ParseExpertShardManifestBody(buffer.str());
}

} // namespace us4
