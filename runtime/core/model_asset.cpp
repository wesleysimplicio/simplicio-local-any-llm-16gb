#include "core/model_asset.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "core/gguf_reader.h"
#include "core/safetensors_reader.h"

namespace us4 {

namespace {

bool WriteError(std::string *error, const std::string &message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

std::string Trim(const std::string &value) {
  const auto begin =
      std::find_if_not(value.begin(), value.end(),
                       [](unsigned char ch) { return std::isspace(ch) != 0; });
  const auto end =
      std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
      }).base();
  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

DType ParseDType(const std::string &value) {
  const std::string normalized = ToLower(value);
  if (normalized == "fp16") {
    return DType::kFloat16;
  }
  if (normalized == "bf16") {
    return DType::kBFloat16;
  }
  if (normalized == "int8") {
    return DType::kInt8;
  }
  if (normalized == "int4") {
    return DType::kInt4;
  }
  return DType::kFloat32;
}

std::vector<std::string> SplitCsv(const std::string &value) {
  std::vector<std::string> parts;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const std::string trimmed = Trim(item);
    if (!trimmed.empty()) {
      parts.push_back(trimmed);
    }
  }
  return parts;
}

bool ParseBool(const std::string &value) {
  const std::string normalized = ToLower(Trim(value));
  return normalized == "true" || normalized == "1" || normalized == "yes";
}

std::vector<std::filesystem::path>
ResolveShardPaths(const std::filesystem::path &baseDirectory,
                  const std::string &value) {
  std::vector<std::filesystem::path> shards;
  for (const std::string &entry : SplitCsv(value)) {
    std::filesystem::path shardPath(entry);
    if (shardPath.is_relative()) {
      shardPath = baseDirectory / shardPath;
    }
    shards.push_back(shardPath.lexically_normal());
  }
  return shards;
}

std::filesystem::path
ResolveSiblingPath(const std::filesystem::path &baseDirectory,
                   const std::string &value) {
  const std::string trimmed = Trim(value);
  if (trimmed.empty()) {
    return {};
  }
  std::filesystem::path resolved(trimmed);
  if (resolved.is_relative()) {
    resolved = baseDirectory / resolved;
  }
  return resolved.lexically_normal();
}

ModelFormat InferFormatFromPath(const std::filesystem::path &path) {
  const std::string extension = ToLower(path.extension().string());
  if (extension == ".us4manifest") {
    return ModelFormat::kFixtureManifest;
  }
  if (extension == ".gguf") {
    return ModelFormat::kGguf;
  }
  if (extension == ".safetensors") {
    return ModelFormat::kSafetensors;
  }
  return ModelFormat::kUnknown;
}

bool LoadFixtureManifest(const std::filesystem::path &path, ModelAsset &asset,
                         std::string *error) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    return WriteError(error, "unable to open model manifest: " + path.string());
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(stream, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    const std::size_t eq = trimmed.find('=');
    if (eq == std::string::npos) {
      return WriteError(error, "invalid manifest line: " + trimmed);
    }

    values.emplace(ToLower(Trim(trimmed.substr(0, eq))),
                   Trim(trimmed.substr(eq + 1)));
  }

  asset.format = ModelFormat::kFixtureManifest;
  asset.family = values["family"];
  asset.modelName = values["model_name"];
  asset.weightDType = ParseDType(values["weight_dtype"]);
  asset.seed = values.contains("seed")
                   ? static_cast<std::uint32_t>(std::stoul(values["seed"]))
                   : 0U;
  asset.vocabulary = SplitCsv(values["vocabulary"]);
  asset.defaultPromptToken = values["default_prompt_token"];
  asset.sourcePath = path;
  asset.sharedTokenizer = values.contains("shared_tokenizer") &&
                          ParseBool(values["shared_tokenizer"]);
  asset.moeLazyLoad =
      values.contains("moe_lazy_load") && ParseBool(values["moe_lazy_load"]);
  asset.moeActiveExperts =
      values.contains("moe_active_experts")
          ? static_cast<std::size_t>(std::stoul(values["moe_active_experts"]))
          : 0U;
  asset.expertShardPaths =
      values.contains("moe_expert_shards")
          ? ResolveShardPaths(path.parent_path(), values["moe_expert_shards"])
          : std::vector<std::filesystem::path>{};
  asset.draftModelPath =
      values.contains("draft_model_path")
          ? ResolveSiblingPath(path.parent_path(), values["draft_model_path"])
          : std::filesystem::path{};
  asset.draftModelFormat = asset.draftModelPath.empty()
                               ? ModelFormat::kUnknown
                               : InferFormatFromPath(asset.draftModelPath);
  asset.metadata = values;

  if (asset.family.empty() || asset.modelName.empty()) {
    return WriteError(error, "manifest must define family and model_name");
  }
  if (asset.vocabulary.empty()) {
    return WriteError(error, "manifest must define vocabulary");
  }
  if (asset.defaultPromptToken.empty()) {
    asset.defaultPromptToken = asset.vocabulary.front();
  }
  return true;
}

void HydrateFromSiblingManifest(const std::filesystem::path &assetPath,
                                ModelAsset &asset) {
  const std::filesystem::path siblingManifest =
      assetPath.parent_path() / "model.us4manifest";
  if (!std::filesystem::exists(siblingManifest)) {
    return;
  }

  ModelAsset manifestAsset;
  std::string ignoredError;
  if (!LoadFixtureManifest(siblingManifest, manifestAsset, &ignoredError)) {
    return;
  }

  if (asset.family.empty()) {
    asset.family = manifestAsset.family;
  }
  asset.weightDType = manifestAsset.weightDType;
  asset.seed = manifestAsset.seed;
  asset.vocabulary = manifestAsset.vocabulary;
  asset.defaultPromptToken = manifestAsset.defaultPromptToken;
  asset.draftModelPath = manifestAsset.draftModelPath;
  asset.draftModelFormat = manifestAsset.draftModelFormat;
  asset.sharedTokenizer = manifestAsset.sharedTokenizer;
  asset.moeLazyLoad = manifestAsset.moeLazyLoad;
  asset.moeActiveExperts = manifestAsset.moeActiveExperts;
  asset.expertShardPaths = manifestAsset.expertShardPaths;
  asset.metadata = manifestAsset.metadata;

  const std::filesystem::path tokenizerPath =
      assetPath.parent_path() / "tokenizer.json";
  if (std::filesystem::exists(tokenizerPath)) {
    asset.metadata["tokenizer_json"] = tokenizerPath.string();
  }
}

std::string InferFamilyFromStem(const std::string &stem) {
  const std::string normalized = ToLower(stem);
  if (normalized.find("qwen") != std::string::npos) {
    return "qwen";
  }
  if (normalized.find("gemma") != std::string::npos) {
    return "gemma";
  }
  if (normalized.find("llama") != std::string::npos) {
    return "llama";
  }
  if (normalized.find("ternary") != std::string::npos) {
    return "ternary";
  }
  if (normalized.find("bitnet") != std::string::npos) {
    return "bitnet";
  }
  if (normalized.find("deepseek") != std::string::npos) {
    return "deepseek";
  }
  if (normalized.find("glm") != std::string::npos) {
    return "glm";
  }
  if (normalized.find("kimi") != std::string::npos) {
    return "kimi";
  }
  return {};
}

} // namespace

std::string_view ToString(const ModelFormat format) {
  switch (format) {
  case ModelFormat::kBuiltin:
    return "builtin";
  case ModelFormat::kFixtureManifest:
    return "fixture-manifest";
  case ModelFormat::kGguf:
    return "gguf";
  case ModelFormat::kSafetensors:
    return "safetensors";
  case ModelFormat::kUnknown:
    return "unknown";
  }
  return "unknown";
}

bool LoadModelAsset(const std::filesystem::path &path, ModelAsset &asset,
                    std::string *error) {
  std::filesystem::path resolved = path;
  if (std::filesystem::is_directory(path)) {
    resolved /= "model.us4manifest";
  }

  const std::string extension = ToLower(resolved.extension().string());
  if (extension == ".us4manifest") {
    return LoadFixtureManifest(resolved, asset, error);
  }

  if (!std::filesystem::exists(resolved)) {
    return WriteError(error,
                      "model asset path does not exist: " + resolved.string());
  }

  asset = {};
  asset.sourcePath = resolved;
  asset.modelName = resolved.stem().string();
  asset.family = InferFamilyFromStem(asset.modelName);
  if (extension == ".gguf") {
    asset.format = ModelFormat::kGguf;
    asset.weightDType = DType::kFloat16;
    HydrateFromSiblingManifest(resolved, asset);

    std::string ggufError;
    if (const auto reader = GgufReader::Open(resolved, &ggufError);
        reader.has_value()) {
      for (const std::string &tensorName :
           {std::string("embedding.weight"), std::string("lm_head.weight")}) {
        std::string readError;
        std::vector<float> values = reader->ReadFloat32(tensorName, &readError);
        if (!values.empty()) {
          if (const auto *info = reader->Find(tensorName); info != nullptr) {
            asset.realTensorShapes[tensorName] = info->shape;
          }
          asset.realTensors[tensorName] = std::move(values);
        }
      }
      asset.hasRealWeights = !asset.realTensors.empty();
      asset.metadata["gguf_load_status"] =
          asset.hasRealWeights ? "real-tensors-loaded"
                               : "real-header-parsed-no-known-tensor-names";
    } else {
      asset.metadata["gguf_load_status"] =
          "placeholder-or-unparseable: " + ggufError;
    }
    return true;
  }
  if (extension == ".safetensors") {
    asset.format = ModelFormat::kSafetensors;
    asset.weightDType = DType::kFloat16;
    HydrateFromSiblingManifest(resolved, asset);

    std::string safetensorsError;
    if (const auto reader =
            SafetensorsReader::Open(resolved, &safetensorsError);
        reader.has_value()) {
      for (const std::string &tensorName :
           {std::string("embedding.weight"), std::string("lm_head.weight")}) {
        std::string readError;
        std::vector<float> values = reader->ReadFloat32(tensorName, &readError);
        if (!values.empty()) {
          if (const auto *info = reader->Find(tensorName); info != nullptr) {
            asset.realTensorShapes[tensorName] = info->shape;
          }
          asset.realTensors[tensorName] = std::move(values);
        }
      }
      asset.hasRealWeights = !asset.realTensors.empty();
      asset.metadata["safetensors_load_status"] =
          asset.hasRealWeights ? "real-tensors-loaded"
                               : "real-header-parsed-no-known-tensor-names";
    } else {
      asset.metadata["safetensors_load_status"] =
          "placeholder-or-unparseable: " + safetensorsError;
    }
    return true;
  }

  return WriteError(error,
                    "unsupported model asset format: " + resolved.string());
}

bool TryLoadExpertShardLmHead(const ModelAsset &asset,
                              const std::size_t expertIndex,
                              const std::size_t vocabSize,
                              std::vector<float> *outWeights,
                              std::vector<std::size_t> *outShape,
                              std::string *error) {
  if (expertIndex >= asset.expertShardPaths.size()) {
    return WriteError(error, "expert index " + std::to_string(expertIndex) +
                                 " has no shard path on this asset");
  }

  const auto reader =
      SafetensorsReader::Open(asset.expertShardPaths[expertIndex], error);
  if (!reader.has_value()) {
    return false;
  }

  const auto *info = reader->Find("lm_head.weight");
  if (info == nullptr || info->shape.size() != 2) {
    return WriteError(error, "expert shard has no real lm_head.weight");
  }
  if (info->shape[0] != vocabSize) {
    return WriteError(
        error, "expert shard lm_head.weight vocab dimension does not match "
               "the requested vocabulary size");
  }

  std::vector<float> weights = reader->ReadFloat32("lm_head.weight", error);
  if (weights.empty()) {
    return false;
  }

  if (outWeights != nullptr) {
    *outWeights = std::move(weights);
  }
  if (outShape != nullptr) {
    *outShape = info->shape;
  }
  return true;
}

namespace {

bool ReadExpertFfnTensor(const SafetensorsReader &reader,
                         const std::string &tensorName,
                         const std::size_t expectedRows,
                         const std::size_t expectedCols,
                         std::vector<float> *outWeights,
                         std::vector<std::size_t> *outShape,
                         std::string *error) {
  const auto *info = reader.Find(tensorName);
  if (info == nullptr || info->shape.size() != 2) {
    return WriteError(error, "expert shard has no real " + tensorName);
  }
  if (info->shape[0] != expectedRows || info->shape[1] != expectedCols) {
    return WriteError(error, "expert shard " + tensorName +
                                 " shape does not match the requested "
                                 "hidden/intermediate size");
  }
  std::vector<float> weights = reader.ReadFloat32(tensorName, error);
  if (weights.empty()) {
    return false;
  }
  if (outWeights != nullptr) {
    *outWeights = std::move(weights);
  }
  if (outShape != nullptr) {
    *outShape = info->shape;
  }
  return true;
}

} // namespace

bool TryLoadExpertShardFfn(const ModelAsset &asset,
                           const std::size_t expertIndex,
                           const std::size_t hiddenSize,
                           const std::size_t intermediateSize,
                           ExpertFfnWeights *outWeights, std::string *error) {
  if (expertIndex >= asset.expertShardPaths.size()) {
    return WriteError(error, "expert index " + std::to_string(expertIndex) +
                                 " has no shard path on this asset");
  }

  const auto reader =
      SafetensorsReader::Open(asset.expertShardPaths[expertIndex], error);
  if (!reader.has_value()) {
    return false;
  }

  ExpertFfnWeights weights;
  // HF/safetensors convention for nn.Linear.weight is [out_features,
  // in_features] -- gate/up project hidden -> intermediate, down projects
  // intermediate -> hidden.
  if (!ReadExpertFfnTensor(*reader, "gate_proj.weight", intermediateSize,
                           hiddenSize, &weights.gate, &weights.gateShape,
                           error) ||
      !ReadExpertFfnTensor(*reader, "up_proj.weight", intermediateSize,
                           hiddenSize, &weights.up, &weights.upShape, error) ||
      !ReadExpertFfnTensor(*reader, "down_proj.weight", hiddenSize,
                           intermediateSize, &weights.down, &weights.downShape,
                           error)) {
    return false;
  }

  if (outWeights != nullptr) {
    *outWeights = std::move(weights);
  }
  return true;
}

} // namespace us4
