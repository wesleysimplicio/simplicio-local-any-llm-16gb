#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "core/backend_selector.h"
#include "core/runtime_context.h"

namespace us4 {

struct ModelAsset;
struct ExpertFfnWeights;

enum class ArchitectureType {
  kDense,
  kMoe,
  kTernary,
  kUnknown,
};

struct GenerationRequest {
  std::string prompt;
  std::size_t maxTokens = 16;
  const ModelAsset *asset = nullptr;
  std::optional<BackendType> requestedBackend = std::nullopt;
  // OpenAI-compatible deterministic override. When absent, the adapter or
  // model asset seed remains authoritative.
  std::optional<std::uint32_t> seed = std::nullopt;
  // Cooperative cancellation for speculative work. The authoritative
  // non-speculative generation remains the caller's source of truth.
  std::stop_token stopToken{};
  // When set (see #81.7c), DenseAdapterBase::Generate routes the attention
  // context through this expert's real FFN (SwiGLU: down(silu(gate(x)) *
  // up(x))) before the output projection, instead of projecting the raw
  // attention context directly.
  const ExpertFfnWeights *expertFfn = nullptr;
};

struct GenerationResult {
  std::string family;
  std::string modelName;
  std::string assetFormat;
  std::string assetPath;
  std::string draftModelFormat;
  std::string draftModelPath;
  std::string speculativeStrategy;
  std::string speculativeSessionScope;
  std::string backend;
  std::string backendReason;
  std::vector<std::string> promptTokens;
  std::vector<std::string> generatedTokens;
  std::string text;
  std::size_t sharedAllocations = 0;
  std::size_t metalDispatches = 0;
  std::size_t mlxOperationCount = 0;
  bool kvCacheHit = false;
  bool kvRestoredFromColdStore = false;
  std::size_t kvPageCount = 0;
  std::size_t kvHotPages = 0;
  std::size_t kvWarmPages = 0;
  std::size_t kvColdPages = 0;
  std::size_t kvSummaryRows = 0;
  std::size_t prefixCacheEntries = 0;
  bool mlxPlanBuilt = false;
  bool mlxEvaluated = false;
  std::size_t moeSelectedExperts = 0;
  float moeRouterEntropy = 0.0F;
  float moeLoadBalance = 0.0F;
  float moeSelectedMass = 0.0F;
  std::size_t moePagerLoads = 0;
  std::size_t moePagerEvictions = 0;
  std::size_t moePagerReuses = 0;
  std::size_t moeResidentExperts = 0;
  std::size_t moeLearnedPinnedExperts = 0;
  std::size_t moePinPromotions = 0;
  std::size_t moePrefetchPrefetched = 0;
  std::size_t moePrefetchHits = 0;
  std::size_t moePrefetchMisses = 0;
  double moePrefetchHitRatio = 0.0;
  bool moePrefetchWrongExpertLeakPrevented = false;
  std::size_t moePrefetchExecutableExperts = 0;
  bool moeSparsityCacheHit = false;
  std::size_t moeSparsityCacheHits = 0;
  std::size_t moeSparsityCacheMisses = 0;
  std::size_t moeSparsityCacheEntries = 0;
  std::size_t moeSparsityWarmEntries = 0;
  double moeSparsityCacheHitRatio = 0.0;
  std::size_t moeSparsityPatternHash = 0;
  std::string moeSparsityPatternKey;
  bool multimodalCacheHit = false;
  std::size_t multimodalCacheHits = 0;
  std::size_t multimodalCacheMisses = 0;
  std::size_t multimodalCacheEntries = 0;
  double multimodalCacheHitRatio = 0.0;
  std::size_t multimodalActiveModalities = 0;
  std::string multimodalModalities;
  std::size_t moeShardCount = 0;
  std::size_t moeActiveExperts = 0;
  bool moeLazyLoad = false;
  bool sharedTokenizer = false;
  bool usedRealWeights = false;
  bool usedRealBpeTokenizer = false;
  std::string tokenizerFallbackReason;
  std::size_t speculativeAcceptedTokens = 0;
  std::size_t speculativeRejectedTokens = 0;
  std::size_t speculativeLookaheadTokens = 0;
  std::size_t speculativeVerifyWindow = 0;
  double speculativeAcceptanceRate = 0.0;
  std::string speculativeFallbackToken;
  bool speculativeWarmupActive = false;
  bool speculativeMtpEnabled = false;
  bool speculativeCancelled = false;
  std::size_t speculativeRounds = 0;
  std::size_t speculativeCommittedTokens = 0;
  std::string speculativeStopReason;
  bool usedRealDraftModel = false;
  bool usedRealExpertWeights = false;
  // Issue #81.7c: distinct from usedRealExpertWeights (which only covers
  // the shared output projection) -- true only when the attention context
  // was actually routed through the selected expert's real FFN layer
  // (gate/up/down_proj), not just a swapped lm_head.
  bool usedRealExpertFfn = false;
  std::string weightDType;
  std::string neonKernelFlavor;
  std::string dequantPath;
  std::string metalDevice;
  std::string metalQueueLabel;
  std::string mixedDispatchStrategy;
  std::size_t mixedDispatchMetalStages = 0;
  std::size_t mixedDispatchAneStages = 0;
  std::size_t aneCompiledLayers = 0;
  std::size_t anePredictionCalls = 0;
  std::string thermalPressureLevel;
  std::string thermalReason;
  bool thermalDowngraded = false;
  RuntimeMode mode = RuntimeMode::kNano;
  bool fellBack = false;
};

class IUS4V6Adapter {
public:
  virtual ~IUS4V6Adapter() = default;

  virtual std::string_view Family() const = 0;
  virtual std::string_view ModelName() const = 0;
  virtual ArchitectureType Architecture() const = 0;

  virtual bool SupportsMoe() const = 0;
  virtual bool SupportsMlxBackend() const = 0;
  virtual bool SupportsMetalBackend() const { return false; }
  virtual bool SupportsAneBackend() const { return false; }
  virtual bool SupportsSpeculativeDecoding() const = 0;
  virtual bool SupportsPromptRun() const = 0;

  virtual RuntimeMode MinimumMode() const = 0;
  virtual RuntimeMode
  RecommendedMode(const HardwareProbeResult &hardware) const = 0;
  virtual void ConfigureRuntime(RuntimeContext &context) const = 0;
  virtual std::vector<std::string> Tokenize(std::string_view text) const = 0;
  virtual GenerationResult Generate(const GenerationRequest &request,
                                    const RuntimeContext &context) const = 0;
};

} // namespace us4
