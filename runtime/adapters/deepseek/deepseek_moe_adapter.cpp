#include "adapters/deepseek/deepseek_moe_adapter.h"

#include <algorithm>
#include <cctype>
#include <sstream>

#include "core/model_asset.h"
#include "moe/speculative_prefetch.h"

namespace us4 {

namespace {

std::string NormalizeRouteToken(std::string token) {
  std::transform(token.begin(), token.end(), token.begin(),
                 [](const unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return token;
}

} // namespace

DeepSeekMoEAdapter::DeepSeekMoEAdapter()
    : DenseAdapterBase("deepseek", "deepseek-v2-lite") {}

ArchitectureType DeepSeekMoEAdapter::Architecture() const {
  return ArchitectureType::kMoe;
}

bool DeepSeekMoEAdapter::SupportsMoe() const { return true; }

bool DeepSeekMoEAdapter::SupportsMlxBackend() const { return true; }

bool DeepSeekMoEAdapter::SupportsMetalBackend() const { return true; }

GenerationResult
DeepSeekMoEAdapter::Generate(const GenerationRequest &request,
                             const RuntimeContext &context) const {
  RuntimeContext &mutableContext = const_cast<RuntimeContext &>(context);
  const std::vector<float> routeLogits =
      BuildRouteLogits(request, request.asset);
  const RouterDecision routing =
      mutableContext.router().RouteTopK(routeLogits, 2);
  const SpeculativePrefetch prefetch(3);
  const SpeculativePrefetchPlan prefetchPlan = prefetch.BuildPlan(
      "deepseek", mutableContext.router().RouteTopK(routeLogits, 3));
  const SpeculativePrefetchTelemetry prefetchTelemetry =
      prefetch.Reconcile(prefetchPlan, routing);
  const SparsityCacheSnapshot cacheSnapshot =
      mutableContext.sparsityCache().Touch("deepseek", routing);
  for (const ExpertScore &expert : routing.selected) {
    const std::string expertId =
        "deepseek-expert-" + std::to_string(expert.expert);
    RecordExpertCacheLookup(mutableContext.adaptiveSpeculativeState(),
                            mutableContext.expertPager().IsResident(expertId));
    mutableContext.expertPager().Touch(expertId);
  }
  const ExpertPagerSnapshot pagerSnapshot =
      mutableContext.expertPager().Snapshot();
  GenerationRequest routedRequest = request;
  const std::string routeSignature = BuildRouteSignature(routing);
  routedRequest.prompt = request.prompt.empty()
                             ? routeSignature
                             : request.prompt + " " + routeSignature;

  // When the routed expert has a genuine shard with a real lm_head.weight
  // (see #81.7), apply that expert's ACTUAL weight to the forward instead
  // of only recording the touch for pager telemetry -- the router
  // decision then really does change what the model computes, not just
  // what gets logged.
  ModelAsset expertAsset;
  ExpertFfnWeights expertFfnWeights;
  bool usedRealExpertWeights = false;
  if (request.asset != nullptr && !routing.selected.empty()) {
    const std::vector<std::string> vocabulary =
        !request.asset->vocabulary.empty() ? request.asset->vocabulary
                                           : Vocabulary();
    std::vector<float> expertLmHead;
    std::vector<std::size_t> expertShape;
    if (TryLoadExpertShardLmHead(
            *request.asset, routing.selected.front().expert, vocabulary.size(),
            &expertLmHead, &expertShape)) {
      expertAsset = *request.asset;
      expertAsset.realTensors["lm_head.weight"] = std::move(expertLmHead);
      expertAsset.realTensorShapes["lm_head.weight"] = expertShape;
      expertAsset.hasRealWeights = true;
      routedRequest.asset = &expertAsset;
      usedRealExpertWeights = true;
    }

    // Issue #81.7c: beyond swapping the shared lm_head.weight above, also
    // try to route the attention context through the SAME expert's real
    // FFN layer (gate/up/down_proj), not just the output projection.
    // kExpertHiddenSize/kExpertIntermediateSize mirror the fixed toy
    // scaffold hidden size DenseAdapterBase::Generate uses internally.
    constexpr std::size_t kExpertHiddenSize = 8;
    constexpr std::size_t kExpertIntermediateSize = 16;
    if (TryLoadExpertShardFfn(*request.asset, routing.selected.front().expert,
                              kExpertHiddenSize, kExpertIntermediateSize,
                              &expertFfnWeights)) {
      routedRequest.expertFfn = &expertFfnWeights;
    }
  }

  GenerationResult result = DenseAdapterBase::Generate(routedRequest, context);
  result.usedRealExpertWeights = usedRealExpertWeights;
  result.family = "deepseek";
  result.text = routeSignature + " " + result.text;
  result.moeSelectedExperts = routing.selected.size();
  result.moeRouterEntropy = routing.entropy;
  result.moeLoadBalance = routing.loadBalance;
  result.moeSelectedMass = routing.selectedMass;
  result.moePagerLoads = pagerSnapshot.loadCount;
  result.moePagerEvictions = pagerSnapshot.evictionCount;
  result.moePagerReuses = pagerSnapshot.reuseCount;
  result.moeResidentExperts = pagerSnapshot.residentCount;
  result.moeLearnedPinnedExperts = pagerSnapshot.learnedPinCount;
  result.moePinPromotions = pagerSnapshot.pinPromotionCount;
  result.moePrefetchPrefetched = prefetchTelemetry.prefetchedCount;
  result.moePrefetchHits = prefetchTelemetry.hitCount;
  result.moePrefetchMisses = prefetchTelemetry.missCount;
  result.moePrefetchHitRatio = prefetchTelemetry.hitRatio;
  result.moePrefetchWrongExpertLeakPrevented =
      prefetchTelemetry.wrongExpertLeakPrevented;
  result.moePrefetchExecutableExperts =
      prefetchTelemetry.executableExperts.size();
  result.moeSparsityCacheHit = cacheSnapshot.lastLookupHit;
  result.moeSparsityCacheHits = cacheSnapshot.hitCount;
  result.moeSparsityCacheMisses = cacheSnapshot.missCount;
  result.moeSparsityCacheEntries = cacheSnapshot.entryCount;
  result.moeSparsityWarmEntries = cacheSnapshot.warmEntryCount;
  result.moeSparsityCacheHitRatio = cacheSnapshot.hitRatio;
  result.moeSparsityPatternHash = cacheSnapshot.lastPatternHash;
  result.moeSparsityPatternKey = cacheSnapshot.lastKey;
  return result;
}

std::uint32_t DeepSeekMoEAdapter::Seed() const { return 52002U; }

std::vector<std::string> DeepSeekMoEAdapter::Vocabulary() const {
  return {"deepseek", "moe",   "experts", "route",  "local", "apple",
          "runtime",  "pager", "selects", "tokens", "with",  "balance",
          ".",        "fast",  "wide",    "hi"};
}

std::string DeepSeekMoEAdapter::DefaultPromptToken() const { return "hi"; }

std::vector<float>
DeepSeekMoEAdapter::BuildRouteLogits(const GenerationRequest &request,
                                     const ModelAsset *asset) const {
  std::vector<float> logits = {0.85F, 0.55F, 0.45F, 0.25F};
  std::vector<std::string> tokens = Tokenize(request.prompt);
  if (tokens.empty() && asset != nullptr &&
      !asset->defaultPromptToken.empty()) {
    tokens.push_back(asset->defaultPromptToken);
  }

  for (const std::string &token : tokens) {
    const std::string normalized = NormalizeRouteToken(token);
    if (normalized.find("code") != std::string::npos ||
        normalized.find("logic") != std::string::npos) {
      logits[0] += 0.35F;
      logits[2] += 0.10F;
    }
    if (normalized.find("math") != std::string::npos ||
        normalized.find("reason") != std::string::npos) {
      logits[1] += 0.30F;
    }
    if (normalized.find("local") != std::string::npos ||
        normalized.find("runtime") != std::string::npos) {
      logits[2] += 0.25F;
    }
    if (normalized.find("wide") != std::string::npos ||
        normalized.find("context") != std::string::npos) {
      logits[3] += 0.20F;
    }
  }

  if (asset != nullptr && asset->metadata.contains("seed")) {
    logits[0] += static_cast<float>(asset->seed % 5U) * 0.01F;
    logits[1] += static_cast<float>(asset->seed % 3U) * 0.01F;
  }

  return logits;
}

std::string
DeepSeekMoEAdapter::BuildRouteSignature(const RouterDecision &routing) const {
  std::ostringstream signature;
  signature << "moe-route";
  for (const ExpertScore &expert : routing.selected) {
    signature << " e" << expert.expert;
  }
  return signature.str();
}

} // namespace us4
