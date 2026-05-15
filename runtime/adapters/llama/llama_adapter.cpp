#include "adapters/llama/llama_adapter.h"
#include "adapters/llama/llama_config.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/backend_selector.h"
#include "core/gqa_attention.h"
#include "core/model_asset.h"
#include "core/rope.h"
#include "core/tensor.h"
#include "cpu/scalar_matmul.h"
#include "neon/kernel_profile.h"
#include "neon/neon_matmul.h"

namespace us4 {

namespace {

std::string NormalizeToken(const std::string_view token) {
  std::string normalized;
  normalized.reserve(token.size());
  for (const char ch : token) {
    if (std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '-' ||
        ch == '_' || ch == '.') {
      normalized.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
  }
  return normalized;
}

std::string BuildPromptCacheKey(const std::uint32_t seed,
                                const std::vector<std::string> &promptTokens) {
  std::ostringstream stream;
  stream << "llama:" << seed << ":";
  for (const std::string &token : promptTokens) {
    stream << NormalizeToken(token) << '|';
  }
  return "kv:llama:" + std::to_string(std::hash<std::string>{}(stream.str()));
}

void CopyVectorToTensor(const std::vector<float> &source, Tensor &tensor) {
  float *target = tensor.MutableDataAsFloat32();
  for (std::size_t index = 0; index < source.size(); ++index) {
    target[index] = source[index];
  }
}

std::vector<float> ReadTensorRow(const Tensor &tensor) {
  const float *source = tensor.DataAsFloat32();
  return {source, source + tensor.ElementCount()};
}

std::string ResolveDequantPath(const ModelAsset *asset) {
  if (asset == nullptr) {
    return "none";
  }
  switch (asset->weightDType) {
  case DType::kInt8:
    return "groupwise-int8";
  case DType::kInt4:
    return "groupwise-int4";
  default:
    return "none";
  }
}

Tensor BuildProjectionTensor(const std::vector<float> &values,
                             const std::vector<std::size_t> &shape,
                             const ModelAsset *asset) {
  const DType dtype = asset == nullptr ? DType::kFloat32 : asset->weightDType;
  if (dtype == DType::kFloat16 || dtype == DType::kBFloat16) {
    Tensor projection(shape, dtype);
    std::uint16_t *data = projection.MutableDataAsUInt16();
    for (std::size_t index = 0; index < values.size(); ++index) {
      data[index] = dtype == DType::kFloat16 ? EncodeFloat16(values[index])
                                             : EncodeBFloat16(values[index]);
    }
    return projection;
  }

  Tensor projection(shape, DType::kFloat32);
  CopyVectorToTensor(values, projection);
  return projection;
}

void DecorateLlamaResult(GenerationResult &result) {
  result.family = "llama";
  if (!result.generatedTokens.empty()) {
    result.generatedTokens[0] = "llama";
  }
  if (!result.text.empty()) {
    result.text = "llama " + result.text;
  }
}

} // namespace

LlamaAdapter::LlamaAdapter() : DenseAdapterBase("llama", "llama-3.1-8b") {}

bool LlamaAdapter::SupportsMlxBackend() const { return true; }

bool LlamaAdapter::SupportsMetalBackend() const { return true; }

std::vector<float>
LlamaAdapter::BuildQueryRow(const std::size_t tokenId, const std::uint32_t seed,
                            const std::size_t position,
                            const LlamaConfig &config) const {
  Tensor row({1, config.hiddenSize}, DType::kFloat32);
  CopyVectorToTensor(BuildTokenEmbedding(tokenId, config.hiddenSize, seed),
                     row);
  ApplyRopeInPlace(row, position, config.ropeTheta, config.ropeScaling,
                   config.ropeScale);
  return ReadTensorRow(row);
}

std::vector<float> LlamaAdapter::BuildKeyRow(const std::size_t tokenId,
                                             const std::uint32_t seed,
                                             const std::size_t position,
                                             const LlamaConfig &config) const {
  const std::size_t kvWidth = config.kvHeads * config.headDim;
  Tensor row({1, kvWidth}, DType::kFloat32);
  CopyVectorToTensor(BuildTokenEmbedding(tokenId, kvWidth, seed + 11U), row);
  ApplyRopeInPlace(row, position, config.ropeTheta, config.ropeScaling,
                   config.ropeScale);
  return ReadTensorRow(row);
}

std::vector<float>
LlamaAdapter::BuildValueRow(const std::size_t tokenId, const std::uint32_t seed,
                            const std::size_t position,
                            const LlamaConfig &config) const {
  const std::size_t kvWidth = config.kvHeads * config.headDim;
  std::vector<float> row = BuildTokenEmbedding(tokenId, kvWidth, seed + 29U);
  for (std::size_t hidden = 0; hidden < row.size(); ++hidden) {
    row[hidden] += static_cast<float>((position + hidden) % 5U) * 0.01F;
  }
  return row;
}

GenerationResult LlamaAdapter::Generate(const GenerationRequest &request,
                                        const RuntimeContext &context) const {
  const BackendSelection backendSelection = SelectBackend(
      context.hardware(), context.mode(), *this, request.requestedBackend);
  if (backendSelection.selected != BackendType::kNeon) {
    GenerationResult result = DenseAdapterBase::Generate(request, context);
    DecorateLlamaResult(result);
    return result;
  }

  RuntimeContext &mutableContext = const_cast<RuntimeContext &>(context);
  mutableContext.SetBackend(backendSelection.selected);

  const std::vector<std::string> vocabulary =
      (request.asset != nullptr && !request.asset->vocabulary.empty())
          ? request.asset->vocabulary
          : Vocabulary();
  const std::uint32_t activeSeed =
      (request.asset != nullptr && request.asset->seed != 0U)
          ? request.asset->seed
          : Seed();
  const LlamaConfig config = ResolveLlamaConfig(request.asset);
  const std::size_t kvWidth = config.kvHeads * config.headDim;

  std::vector<std::string> promptTokens = Tokenize(request.prompt);
  if (promptTokens.empty()) {
    if (request.asset != nullptr &&
        !request.asset->defaultPromptToken.empty()) {
      promptTokens.push_back(request.asset->defaultPromptToken);
    } else {
      promptTokens.push_back(DefaultPromptToken());
    }
  }

  std::vector<std::size_t> tokenIds;
  tokenIds.reserve(promptTokens.size() + request.maxTokens);
  for (const std::string &token : promptTokens) {
    tokenIds.push_back(TokenIdFor(token, vocabulary));
  }

  const std::string prefixKey = BuildPromptCacheKey(activeSeed, promptTokens);
  mutableContext.prefixCache().Retain(prefixKey);

  std::vector<float> keyBuffer;
  std::vector<float> valueBuffer;
  bool kvCacheHit = false;
  if (const std::optional<KvPage> cachedPage =
          mutableContext.kvPager().Lookup(prefixKey);
      cachedPage.has_value() && cachedPage->rowWidth == kvWidth &&
      cachedPage->rowCount == promptTokens.size() &&
      cachedPage->keys.size() == cachedPage->values.size()) {
    keyBuffer = cachedPage->keys;
    valueBuffer = cachedPage->values;
    kvCacheHit = true;
  } else {
    const std::vector<float> firstKeyRow =
        BuildKeyRow(tokenIds.front(), activeSeed, 0U, config);
    const std::vector<float> firstValueRow =
        BuildValueRow(tokenIds.front(), activeSeed, 0U, config);
    mutableContext.kvPager().Append(prefixKey, firstKeyRow, firstValueRow,
                                    kvWidth);
    keyBuffer = firstKeyRow;
    valueBuffer = firstValueRow;

    for (std::size_t index = 1; index < tokenIds.size(); ++index) {
      const std::vector<float> keyRow =
          BuildKeyRow(tokenIds[index], activeSeed, index, config);
      const std::vector<float> valueRow =
          BuildValueRow(tokenIds[index], activeSeed, index, config);
      const bool appended =
          mutableContext.kvPager().AppendRow(prefixKey, keyRow, valueRow);
      if (!appended) {
        std::vector<float> mergedKeys = keyBuffer;
        std::vector<float> mergedValues = valueBuffer;
        mergedKeys.insert(mergedKeys.end(), keyRow.begin(), keyRow.end());
        mergedValues.insert(mergedValues.end(), valueRow.begin(),
                            valueRow.end());
        mutableContext.kvPager().Append(prefixKey, std::move(mergedKeys),
                                        std::move(mergedValues), kvWidth);
      }
      keyBuffer.insert(keyBuffer.end(), keyRow.begin(), keyRow.end());
      valueBuffer.insert(valueBuffer.end(), valueRow.begin(), valueRow.end());
    }
  }

  std::vector<std::string> generatedTokens;
  generatedTokens.reserve(request.maxTokens);
  for (std::size_t step = 0; step < request.maxTokens; ++step) {
    const std::size_t sequenceLength = keyBuffer.size() / kvWidth;
    Tensor key({sequenceLength, kvWidth}, DType::kFloat32);
    Tensor value({sequenceLength, kvWidth}, DType::kFloat32);
    Tensor query({1, config.hiddenSize}, DType::kFloat32);
    Tensor contextTensor({1, config.hiddenSize}, DType::kFloat32);
    Tensor logits({1, vocabulary.size()}, DType::kFloat32);

    CopyVectorToTensor(keyBuffer, key);
    CopyVectorToTensor(valueBuffer, value);
    CopyVectorToTensor(
        BuildQueryRow(tokenIds.back(), activeSeed, sequenceLength - 1U, config),
        query);

    std::string error;
    if (!GqaAttention(query, key, value, config.queryHeads, config.kvHeads,
                      contextTensor, &error)) {
      generatedTokens.push_back("gqa-error");
      break;
    }

    const ModelAsset *projectionAsset =
        backendSelection.selected == BackendType::kNeon ? request.asset
                                                        : nullptr;
    const Tensor projection = BuildProjectionTensor(
        BuildOutputProjection(vocabulary, config.hiddenSize, activeSeed),
        {config.hiddenSize, vocabulary.size()}, projectionAsset);
    const bool matmulOk =
        backendSelection.selected == BackendType::kNeon
            ? NeonMatmul(contextTensor, projection, logits, &error)
            : ScalarMatmul(contextTensor, projection, logits, &error);
    if (!matmulOk) {
      generatedTokens.push_back("matmul-error");
      break;
    }

    const float *logitData = logits.DataAsFloat32();
    std::size_t bestIndex = 0U;
    float bestValue = logitData[0];
    for (std::size_t index = 1; index < vocabulary.size(); ++index) {
      const float bias =
          static_cast<float>(((step + 1U) * (index + 5U)) % 9U) * 0.003F;
      const float candidate = logitData[index] + bias;
      if (candidate > bestValue) {
        bestValue = candidate;
        bestIndex = index;
      }
    }

    generatedTokens.push_back(vocabulary[bestIndex]);
    tokenIds.push_back(bestIndex);

    const std::vector<float> nextKeyRow =
        BuildKeyRow(bestIndex, activeSeed, sequenceLength, config);
    const std::vector<float> nextValueRow =
        BuildValueRow(bestIndex, activeSeed, sequenceLength, config);
    keyBuffer.insert(keyBuffer.end(), nextKeyRow.begin(), nextKeyRow.end());
    valueBuffer.insert(valueBuffer.end(), nextValueRow.begin(),
                       nextValueRow.end());
  }

  GenerationResult result;
  result.family = (request.asset != nullptr && !request.asset->family.empty())
                      ? request.asset->family
                      : "llama";
  result.modelName =
      (request.asset != nullptr && !request.asset->modelName.empty())
          ? request.asset->modelName
          : std::string(ModelName());
  result.assetFormat = request.asset != nullptr
                           ? std::string(ToString(request.asset->format))
                           : "builtin";
  result.assetPath =
      request.asset != nullptr ? request.asset->sourcePath.string() : "";
  result.backend = std::string(ToString(backendSelection.selected));
  result.backendReason = std::string(backendSelection.reason);
  result.promptTokens = std::move(promptTokens);
  result.generatedTokens = generatedTokens;
  result.text = JoinTokens(generatedTokens);
  result.sharedAllocations = context.allocator().SharedAllocationCount();
  result.metalDispatches = context.metalQueue().DispatchCount();
  result.mlxOperationCount =
      context.mlxBridge().LastPlan().has_value()
          ? context.mlxBridge().LastPlan()->operations.size()
          : 0U;
  result.kvCacheHit = kvCacheHit;
  result.kvRestoredFromColdStore = false;
  result.kvPageCount = mutableContext.kvPager().PageCount();
  result.kvHotPages = mutableContext.kvPager().HotPageCount();
  result.kvWarmPages = mutableContext.kvPager().WarmPageCount();
  result.kvColdPages = mutableContext.kvPager().ColdPageCount();
  result.kvSummaryRows = 0U;
  result.prefixCacheEntries = mutableContext.prefixCache().EntryCount();
  result.mlxPlanBuilt = context.mlxBridge().LastPlan().has_value();
  result.mlxEvaluated = context.mlxBridge().LastEvaluationSucceeded();
  result.weightDType = request.asset != nullptr
                           ? std::string(ToString(request.asset->weightDType))
                           : "fp32";
  result.dequantPath = ResolveDequantPath(request.asset);
  result.neonKernelFlavor = "none";
  if (backendSelection.selected == BackendType::kNeon) {
    const DType planDType =
        request.asset != nullptr ? request.asset->weightDType : DType::kFloat32;
    const Tensor lhs(
        {std::max<std::size_t>(request.maxTokens, 1U), config.hiddenSize},
        planDType, DeviceType::kCpu);
    const Tensor rhs({config.hiddenSize, config.hiddenSize}, planDType,
                     DeviceType::kCpu);
    result.neonKernelFlavor = std::string(
        ToString(PlanNeonMatmul(context.hardware(), lhs, rhs).flavor));
  }
  result.metalDevice = context.metalQueue().Device().deviceName;
  result.metalQueueLabel = context.metalQueue().Device().queueLabel;
  result.mode = context.mode();
  result.fellBack = backendSelection.fellBack;
  DecorateLlamaResult(result);
  return result;
}

std::uint32_t LlamaAdapter::Seed() const { return 31800U; }

std::vector<std::string> LlamaAdapter::Vocabulary() const {
  return {"llama", "apple",  "runtime", "dense",  "adapter", "gqa",
          "rope",  "metal",  "local",   "tokens", "reply",   "hello",
          ".",     "steady", "wide",    "context"};
}

std::string LlamaAdapter::DefaultPromptToken() const { return "hello"; }

} // namespace us4
