#include "adapters/kimi/kimi_moe_adapter.h"

namespace us4 {

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
  const RouterDecision routing =
      mutableContext.router().RouteTopK({0.5F, 0.8F, 0.3F, 0.7F}, 2);
  for (const ExpertScore &expert : routing.selected) {
    mutableContext.expertPager().Touch("kimi-expert-" +
                                       std::to_string(expert.expert));
  }
  const ExpertPagerSnapshot pagerSnapshot =
      mutableContext.expertPager().Snapshot();
  GenerationResult result = DenseAdapterBase::Generate(request, context);
  result.family = "kimi";
  result.text = "kimi " + result.text;
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

std::uint32_t KimiMoEAdapter::Seed() const { return 62002U; }

std::vector<std::string> KimiMoEAdapter::Vocabulary() const {
  return {"kimi",  "moe",     "routing", "experts",   "local",    "context",
          "apple", "runtime", "token",   "selection", "responds", "now",
          ".",     "smart",   "fast",    "hi"};
}

std::string KimiMoEAdapter::DefaultPromptToken() const { return "hi"; }

} // namespace us4
