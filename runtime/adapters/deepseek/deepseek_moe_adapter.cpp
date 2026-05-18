#include "adapters/deepseek/deepseek_moe_adapter.h"

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
  const RouterDecision routing = mutableContext.router().RouteTopK(
      BuildRouteLogits(request, request.asset), 2);
  for (const ExpertScore &expert : routing.selected) {
    mutableContext.expertPager().Touch("deepseek-expert-" +
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
