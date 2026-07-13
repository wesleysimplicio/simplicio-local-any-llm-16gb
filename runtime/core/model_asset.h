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
  std::string chatTemplate;
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

// Reads the real "lm_head.weight" tensor from the shard file at
// `asset.expertShardPaths[expertIndex]` (a genuine .safetensors file, parsed
// via SafetensorsReader -- not the shard manifest's text-only bookkeeping),
// so a router decision can drive which expert's ACTUAL weight the forward
// uses instead of only recording that the expert was "touched" for
// telemetry. Returns false (with `error` set) when the shard is missing,
// unreadable, or its shape doesn't match `vocabSize` under the real HF
// convention ([vocab_size, hidden_size]).
bool TryLoadExpertShardLmHead(const ModelAsset &asset, std::size_t expertIndex,
                              std::size_t vocabSize,
                              std::vector<float> *outWeights,
                              std::vector<std::size_t> *outShape,
                              std::string *error = nullptr);

// The real per-expert MLP ("FFN") weights for a Mixtral/DeepSeek-style MoE
// expert: gate_proj/up_proj project hidden -> intermediate, down_proj
// projects back intermediate -> hidden (see ApplyExpertFfnSwiglu in
// runtime/moe/expert_ffn.h for how these combine into the SwiGLU forward).
struct ExpertFfnWeights {
  std::vector<float> gate;
  std::vector<std::size_t> gateShape;
  std::vector<float> up;
  std::vector<std::size_t> upShape;
  std::vector<float> down;
  std::vector<std::size_t> downShape;
};

// Reads the real "gate_proj.weight"/"up_proj.weight"/"down_proj.weight"
// tensors from the shard file at `asset.expertShardPaths[expertIndex]`, the
// same shard TryLoadExpertShardLmHead reads "lm_head.weight" from. This is
// what lets the router's selected expert route through its own FFN layer
// (see #81.7c), not just swap the shared output projection (#81.7/#81.7b).
// Returns false (with `error` set) when the shard is missing, unreadable,
// or any of the three tensors is absent or shaped incompatibly with
// `hiddenSize`/`intermediateSize`.
bool TryLoadExpertShardFfn(const ModelAsset &asset, std::size_t expertIndex,
                           std::size_t hiddenSize, std::size_t intermediateSize,
                           ExpertFfnWeights *outWeights,
                           std::string *error = nullptr);

} // namespace us4
