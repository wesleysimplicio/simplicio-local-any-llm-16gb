#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
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
};

std::string_view ToString(ModelFormat format);
bool LoadModelAsset(const std::filesystem::path& path, ModelAsset& asset, std::string* error = nullptr);

}  // namespace us4
