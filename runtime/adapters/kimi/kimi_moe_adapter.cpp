#include "adapters/kimi/kimi_moe_adapter.h"

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

KimiMoEAdapter::KimiMoEAdapter()
    : DenseAdapterBase("kimi", "kimi-k2-instruct") {}

ArchitectureType KimiMoEAdapter::Architecture() const {
  return ArchitectureType::kMoe;
}

bool KimiMoEAdapter::SupportsMoe() const { return true; }

bool KimiMoEAdapter::SupportsMlxBackend() const { return true; }

bool KimiMoEAdapter::SupportsMetalBackend() const { return true; }

GenerationResult KimiMoEAdapter::Generate(const GenerationRequest &request,
                                          const RuntimeContext &context) const {
  RuntimeContext &mutableContext = const_cast<RuntimeContext &>(context);
  const std::vector<float> routeLogits =
      BuildRouteLogits(request, request.asset);
  const RouterDecision routing =
      mutableContext.router().RouteTopK(routeLogits, 2);
  const SpeculativePrefetch prefetch(3);
  const SpeculativePrefetchPlan prefetchPlan = prefetch.BuildPlan(
      "kimi", mutableContext.router().RouteTopK(routeLogits, 3));
  const SpeculativePrefetchTelemetry prefetchTelemetry =
      prefetch.Reconcile(prefetchPlan, routing);
  const SparsityCacheSnapshot cacheSnapshot =
      mutableContext.sparsityCache().Touch("kimi", routing);
  for (const ExpertScore &expert : routing.selected) {
    mutableContext.expertPager().Touch("kimi-expert-" +
                                       std::to_string(expert.expert));
  }
  const ExpertPagerSnapshot pagerSnapshot =
      mutableContext.expertPager().Snapshot();
  GenerationRequest routedRequest = request;
  const std::string routeSignature = BuildRouteSignature(routing);
  routedRequest.prompt = request.prompt.empty()
                             ? routeSignature
                             : request.prompt + " " + routeSignature;

  // When the routed expert has a genuine shard with a real lm_head.weight
  // (see #81.7/#81.7b), apply that expert's ACTUAL weight to the forward
  // instead of only recording the touch for pager telemetry -- the router
  // decision then really does change what the model computes, not just
  // what gets logged.
  ModelAsset expertAsset;
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
  }

  GenerationResult result = DenseAdapterBase::Generate(routedRequest, context);
  result.usedRealExpertWeights = usedRealExpertWeights;
  result.family = "kimi";
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

std::uint32_t KimiMoEAdapter::Seed() const { return 62002U; }

std::vector<std::string> KimiMoEAdapter::Vocabulary() const {
  return {"kimi",  "moe",     "routing", "experts",   "local",    "context",
          "apple", "runtime", "token",   "selection", "responds", "now",
          ".",     "smart",   "fast",    "hi"};
}

std::string KimiMoEAdapter::DefaultPromptToken() const { return "hi"; }

std::vector<float>
KimiMoEAdapter::BuildRouteLogits(const GenerationRequest &request,
                                 const ModelAsset *asset) const {
  std::vector<float> logits = {0.35F, 0.85F, 0.45F, 0.65F};
  std::vector<std::string> tokens = Tokenize(request.prompt);
  if (tokens.empty() && asset != nullptr &&
      !asset->defaultPromptToken.empty()) {
    tokens.push_back(asset->defaultPromptToken);
  }

  for (const std::string &token : tokens) {
    const std::string normalized = NormalizeRouteToken(token);
    if (normalized.find("agent") != std::string::npos ||
        normalized.find("tool") != std::string::npos) {
      logits[0] += 0.30F;
      logits[2] += 0.10F;
    }
    if (normalized.find("smart") != std::string::npos ||
        normalized.find("reason") != std::string::npos) {
      logits[1] += 0.30F;
    }
    if (normalized.find("fast") != std::string::npos ||
        normalized.find("local") != std::string::npos) {
      logits[2] += 0.30F;
    }
    if (normalized.find("wide") != std::string::npos ||
        normalized.find("context") != std::string::npos) {
      logits[3] += 0.25F;
    }
  }

  if (asset != nullptr && asset->metadata.contains("seed")) {
    logits[1] += static_cast<float>(asset->seed % 5U) * 0.01F;
    logits[3] += static_cast<float>(asset->seed % 3U) * 0.01F;
  }

  return logits;
}

std::string
KimiMoEAdapter::BuildRouteSignature(const RouterDecision &routing) const {
  std::ostringstream signature;
  signature << "kimi-route";
  for (const ExpertScore &expert : routing.selected) {
    signature << " e" << expert.expert;
  }
  return signature.str();
}

} // namespace us4
