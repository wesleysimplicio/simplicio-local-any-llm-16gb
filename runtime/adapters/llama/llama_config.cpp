#include "adapters/llama/llama_config.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <string>

namespace us4 {

namespace {

std::size_t ParseSizeOrDefault(const ModelAsset *asset, const char *key,
                               const std::size_t fallback) {
  if (asset == nullptr) {
    return fallback;
  }
  const auto it = asset->metadata.find(key);
  if (it == asset->metadata.end()) {
    return fallback;
  }
  std::size_t value = fallback;
  const char *begin = it->second.data();
  const char *end = begin + it->second.size();
  const auto result = std::from_chars(begin, end, value);
  return result.ec == std::errc{} ? value : fallback;
}

float ParseFloatOrDefault(const ModelAsset *asset, const char *key,
                          const float fallback) {
  if (asset == nullptr) {
    return fallback;
  }
  const auto it = asset->metadata.find(key);
  if (it == asset->metadata.end()) {
    return fallback;
  }
  try {
    return std::stof(it->second);
  } catch (...) {
    return fallback;
  }
}

RopeScalingType ParseRopeScaling(const ModelAsset *asset) {
  if (asset == nullptr) {
    return RopeScalingType::kDynamic;
  }
  const auto it = asset->metadata.find("rope_scaling");
  if (it == asset->metadata.end()) {
    return RopeScalingType::kDynamic;
  }

  std::string value = it->second;
  std::transform(value.begin(), value.end(), value.begin(),
                 [](const unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (value == "linear") {
    return RopeScalingType::kLinear;
  }
  if (value == "yarn") {
    return RopeScalingType::kYaRN;
  }
  return RopeScalingType::kDynamic;
}

} // namespace

LlamaConfig ResolveLlamaConfig(const ModelAsset *asset) {
  LlamaConfig config;
  config.hiddenSize =
      ParseSizeOrDefault(asset, "hidden_size", config.hiddenSize);
  config.queryHeads =
      ParseSizeOrDefault(asset, "query_heads", config.queryHeads);
  config.kvHeads = ParseSizeOrDefault(asset, "kv_heads", config.kvHeads);
  config.headDim = ParseSizeOrDefault(asset, "head_dim", config.headDim);
  config.ropeTheta = ParseFloatOrDefault(asset, "rope_theta", config.ropeTheta);
  config.ropeScaling = ParseRopeScaling(asset);
  config.ropeScale = ParseFloatOrDefault(asset, "rope_scale", config.ropeScale);

  if (config.queryHeads == 0U) {
    config.queryHeads = 2U;
  }
  if (config.kvHeads == 0U || config.kvHeads > config.queryHeads ||
      (config.queryHeads % config.kvHeads) != 0U) {
    config.kvHeads = 1U;
  }
  if (config.headDim == 0U) {
    config.headDim =
        std::max<std::size_t>(1U, config.hiddenSize / config.queryHeads);
  }

  const std::size_t derivedHidden = config.headDim * config.queryHeads;
  if (config.hiddenSize == 0U) {
    config.hiddenSize = derivedHidden;
  } else if (derivedHidden != config.hiddenSize) {
    config.headDim =
        std::max<std::size_t>(1U, config.hiddenSize / config.queryHeads);
    config.hiddenSize = config.headDim * config.queryHeads;
  }

  if (config.ropeTheta <= 1.0F) {
    config.ropeTheta = 10000.0F;
  }
  if (config.ropeScale <= 0.0F) {
    config.ropeScale = 1.0F;
  }

  return config;
}

} // namespace us4
