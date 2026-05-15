#include "adapters/deepseek/deepseek_moe_adapter.h"

namespace us4 {

DeepSeekMoEAdapter::DeepSeekMoEAdapter() : DenseAdapterBase("deepseek", "deepseek-v2-lite") {}

ArchitectureType DeepSeekMoEAdapter::Architecture() const { return ArchitectureType::kMoe; }

bool DeepSeekMoEAdapter::SupportsMoe() const { return true; }

bool DeepSeekMoEAdapter::SupportsMlxBackend() const { return true; }

bool DeepSeekMoEAdapter::SupportsMetalBackend() const { return true; }

GenerationResult DeepSeekMoEAdapter::Generate(const GenerationRequest& request, const RuntimeContext& context) const {
  RuntimeContext& mutableContext = const_cast<RuntimeContext&>(context);
  const auto experts = mutableContext.router().TopK({0.9F, 0.4F, 0.7F, 0.2F}, 2);
  for (const ExpertScore& expert : experts) {
    mutableContext.expertPager().Touch("deepseek-expert-" + std::to_string(expert.expert));
  }
  GenerationResult result = DenseAdapterBase::Generate(request, context);
  result.family = "deepseek";
  result.text = "moe " + result.text;
  return result;
}

std::uint32_t DeepSeekMoEAdapter::Seed() const { return 52002U; }

std::vector<std::string> DeepSeekMoEAdapter::Vocabulary() const {
  return {"deepseek", "moe", "experts", "route", "local", "apple", "runtime", "pager",
          "selects", "tokens", "with", "balance", ".", "fast", "wide", "hi"};
}

std::string DeepSeekMoEAdapter::DefaultPromptToken() const { return "hi"; }

}  // namespace us4
