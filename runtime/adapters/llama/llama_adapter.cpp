#include "adapters/llama/llama_adapter.h"

#include "core/gqa_attention.h"
#include "core/rope.h"

namespace us4 {

LlamaAdapter::LlamaAdapter() : DenseAdapterBase("llama", "llama-3.1-8b") {}

bool LlamaAdapter::SupportsMlxBackend() const { return true; }

bool LlamaAdapter::SupportsMetalBackend() const { return true; }

GenerationResult LlamaAdapter::Generate(const GenerationRequest& request, const RuntimeContext& context) const {
  GenerationResult result = DenseAdapterBase::Generate(request, context);
  result.family = "llama";
  if (!result.generatedTokens.empty()) {
    result.generatedTokens[0] = "llama";
  }
  if (!result.text.empty()) {
    result.text = "llama " + result.text;
  }
  return result;
}

std::uint32_t LlamaAdapter::Seed() const { return 31800U; }

std::vector<std::string> LlamaAdapter::Vocabulary() const {
  return {"llama", "apple", "runtime", "dense", "adapter", "gqa", "rope", "metal",
          "local", "tokens", "reply", "hello", ".", "steady", "wide", "context"};
}

std::string LlamaAdapter::DefaultPromptToken() const { return "hello"; }

}  // namespace us4
