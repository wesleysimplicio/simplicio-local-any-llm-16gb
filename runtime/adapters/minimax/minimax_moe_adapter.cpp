#include "adapters/minimax/minimax_moe_adapter.h"

#include <algorithm>
#include <cctype>
#include <sstream>

#include "core/model_asset.h"

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

MiniMaxMoEAdapter::MiniMaxMoEAdapter()
    : DenseAdapterBase("minimax", "minimax-m2") {}

ArchitectureType MiniMaxMoEAdapter::Architecture() const {
  return ArchitectureType::kMoe;
}

bool MiniMaxMoEAdapter::SupportsMoe() const { return true; }

bool MiniMaxMoEAdapter::SupportsMlxBackend() const { return true; }

bool MiniMaxMoEAdapter::SupportsMetalBackend() const { return true; }

GenerationResult
MiniMaxMoEAdapter::Generate(const GenerationRequest &request,
                            const RuntimeContext &context) const {
  RuntimeContext &mutableContext = const_cast<RuntimeContext &>(context);
  const RouterDecision routing = mutableContext.router().RouteTopK(
      BuildRouteLogits(request, request.asset), 2);
  for (const ExpertScore &expert : routing.selected) {
    mutableContext.expertPager().Touch("minimax-expert-" +
                                       std::to_string(expert.expert));
  }
  const ExpertPagerSnapshot pagerSnapshot =
      mutableContext.expertPager().Snapshot();
  GenerationRequest routedRequest = request;
  const std::string routeSignature = BuildRouteSignature(routing);
  routedRequest.prompt = request.prompt.empty()
                             ? routeSignature
                             : request.prompt + " " + routeSignature;
  GenerationResult result = DenseAdapterBase::Generate(routedRequest, context);
  result.family = "minimax";
  result.text = routeSignature + " " + result.text;
  result.moeSelectedExperts = routing.selected.size();
  result.moeRouterEntropy = routing.entropy;
  result.moeLoadBalance = routing.loadBalance;
  result.moeSelectedMass = routing.selectedMass;
  result.moePagerLoads = pagerSnapshot.loadCount;
  result.moePagerEvictions = pagerSnapshot.evictionCount;
  result.moePagerReuses = pagerSnapshot.reuseCount;
  result.moeResidentExperts = pagerSnapshot.residentCount;
  return result;
}

std::uint32_t MiniMaxMoEAdapter::Seed() const { return 72027U; }

std::vector<std::string> MiniMaxMoEAdapter::Vocabulary() const {
  return {"minimax",  "moe",      "multimodal", "experts", "vision", "audio",
          "fusion",   "routing",  "context",    "local",   "apple",  "runtime",
          "responds", "balanced", "wide",       "hi"};
}

std::string MiniMaxMoEAdapter::DefaultPromptToken() const { return "hi"; }

std::vector<float>
MiniMaxMoEAdapter::BuildRouteLogits(const GenerationRequest &request,
                                    const ModelAsset *asset) const {
  std::vector<float> logits = {0.55F, 0.75F, 0.45F, 0.60F};
  std::vector<std::string> tokens = Tokenize(request.prompt);
  if (tokens.empty() && asset != nullptr &&
      !asset->defaultPromptToken.empty()) {
    tokens.push_back(asset->defaultPromptToken);
  }

  for (const std::string &token : tokens) {
    const std::string normalized = NormalizeRouteToken(token);
    if (normalized.find("image") != std::string::npos ||
        normalized.find("vision") != std::string::npos) {
      logits[0] += 0.35F;
      logits[3] += 0.10F;
    }
    if (normalized.find("audio") != std::string::npos ||
        normalized.find("speech") != std::string::npos) {
      logits[1] += 0.35F;
    }
    if (normalized.find("tool") != std::string::npos ||
        normalized.find("reason") != std::string::npos ||
        normalized.find("logic") != std::string::npos) {
      logits[2] += 0.30F;
    }
    if (normalized.find("context") != std::string::npos ||
        normalized.find("fusion") != std::string::npos ||
        normalized.find("wide") != std::string::npos) {
      logits[3] += 0.30F;
    }
  }

  if (asset != nullptr && asset->metadata.contains("seed")) {
    logits[0] += static_cast<float>(asset->seed % 5U) * 0.01F;
    logits[2] += static_cast<float>(asset->seed % 3U) * 0.01F;
  }

  return logits;
}

std::string
MiniMaxMoEAdapter::BuildRouteSignature(const RouterDecision &routing) const {
  std::ostringstream signature;
  signature << "minimax-route";
  for (const ExpertScore &expert : routing.selected) {
    signature << " e" << expert.expert;
  }
  return signature.str();
}

} // namespace us4
