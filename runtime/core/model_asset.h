#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"

namespace us4 {

enum class ModelFormat {
  kBuiltin,
  kFixtureManifest,
  kGguf,
  kSafetensors,
  kUnknown,
};

struct ModelAsset {
  std::string family;
  std::string modelName;
  ModelFormat format = ModelFormat::kUnknown;
  DType weightDType = DType::kFloat32;
  std::uint32_t seed = 0;
  std::vector<std::string> vocabulary;
  std::string defaultPromptToken;
  std::filesystem::path sourcePath;
  std::filesystem::path draftModelPath;
  ModelFormat draftModelFormat = ModelFormat::kUnknown;
  bool sharedTokenizer = false;
  bool moeLazyLoad = false;
  std::size_t moeActiveExperts = 0;
  std::vector<std::filesystem::path> expertShardPaths;
  std::unordered_map<std::string, std::string> metadata;
  // Populated only when sourcePath points at a real, parseable .safetensors
  // file (genuine header + tensor bytes) rather than a placeholder/manifest.
  // See SafetensorsReader; keys are tensor names as they appear in the file.
  std::unordered_map<std::string, std::vector<float>> realTensors;
  std::unordered_map<std::string, std::vector<std::size_t>> realTensorShapes;
  bool hasRealWeights = false;
};

std::string_view ToString(ModelFormat format);
bool LoadModelAsset(const std::filesystem::path &path, ModelAsset &asset,
                    std::string *error = nullptr);

} // namespace us4
