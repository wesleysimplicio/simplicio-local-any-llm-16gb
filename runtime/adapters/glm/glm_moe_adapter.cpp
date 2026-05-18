#include "adapters/glm/glm_moe_adapter.h"

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

GlmMoEAdapter::GlmMoEAdapter() : DenseAdapterBase("glm", "glm-5.1") {}

ArchitectureType GlmMoEAdapter::Architecture() const {
  return ArchitectureType::kMoe;
}

bool GlmMoEAdapter::SupportsMoe() const { return true; }

bool GlmMoEAdapter::SupportsMlxBackend() const { return true; }

bool GlmMoEAdapter::SupportsMetalBackend() const { return true; }

GenerationResult GlmMoEAdapter::Generate(const GenerationRequest &request,
                                         const RuntimeContext &context) const {
  RuntimeContext &mutableContext = const_cast<RuntimeContext &>(context);
  const RouterDecision routing = mutableContext.router().RouteTopK(
      BuildRouteLogits(request, request.asset), 2);
  for (const ExpertScore &expert : routing.selected) {
    mutableContext.expertPager().Touch("glm-expert-" +
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
  result.family = "glm";
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

std::uint32_t GlmMoEAdapter::Seed() const { return 51045U; }

std::vector<std::string> GlmMoEAdapter::Vocabulary() const {
  return {"glm",     "moe",    "planner", "experts", "reason", "tools",
          "context", "code",   "vision",  "runtime", "local",  "apple",
          "wide",    "assist", "fast",    "hi"};
}

std::string GlmMoEAdapter::DefaultPromptToken() const { return "hi"; }

std::vector<float>
GlmMoEAdapter::BuildRouteLogits(const GenerationRequest &request,
                                const ModelAsset *asset) const {
  std::vector<float> logits = {0.65F, 0.55F, 0.80F, 0.45F};
  std::vector<std::string> tokens = Tokenize(request.prompt);
  if (tokens.empty() && asset != nullptr &&
      !asset->defaultPromptToken.empty()) {
    tokens.push_back(asset->defaultPromptToken);
  }

  for (const std::string &token : tokens) {
    const std::string normalized = NormalizeRouteToken(token);
    if (normalized.find("tool") != std::string::npos ||
        normalized.find("agent") != std::string::npos) {
      logits[0] += 0.30F;
    }
    if (normalized.find("vision") != std::string::npos ||
        normalized.find("image") != std::string::npos) {
      logits[1] += 0.25F;
    }
    if (normalized.find("reason") != std::string::npos ||
        normalized.find("logic") != std::string::npos ||
        normalized.find("code") != std::string::npos) {
      logits[2] += 0.35F;
    }
    if (normalized.find("wide") != std::string::npos ||
        normalized.find("context") != std::string::npos ||
        normalized.find("long") != std::string::npos) {
      logits[3] += 0.30F;
    }
  }

  if (asset != nullptr && asset->metadata.contains("seed")) {
    logits[1] += static_cast<float>(asset->seed % 7U) * 0.01F;
    logits[3] += static_cast<float>(asset->seed % 5U) * 0.01F;
  }

  return logits;
}

std::string
GlmMoEAdapter::BuildRouteSignature(const RouterDecision &routing) const {
  std::ostringstream signature;
  signature << "glm-route";
  for (const ExpertScore &expert : routing.selected) {
    signature << " e" << expert.expert;
  }
  return signature.str();
}

} // namespace us4
