#include "adapters/adapter_registry.h"

#include <array>
#include <cctype>
#include <string>

#include "adapters/bitnet/bitnet_adapter.h"
#include "adapters/deepseek/deepseek_moe_adapter.h"
#include "adapters/gemma/gemma_adapter.h"
#include "adapters/glm/glm_moe_adapter.h"
#include "adapters/kimi/kimi_moe_adapter.h"
#include "adapters/llama/llama_adapter.h"
#include "adapters/minimax/minimax_moe_adapter.h"
#include "adapters/qwen/qwen_adapter.h"
#include "adapters/ternary/ternary_adapter.h"

namespace us4 {

namespace {

std::string Normalize(const std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (const char ch : value) {
    normalized.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return normalized;
}

const std::array<const IUS4V6Adapter *, 9> &AdapterTable() {
  static const BitNetAdapter kBitNetAdapter;
  static const DeepSeekMoEAdapter kDeepSeekAdapter;
  static const QwenAdapter kQwenAdapter;
  static const GemmaAdapter kGemmaAdapter;
  static const GlmMoEAdapter kGlmAdapter;
  static const KimiMoEAdapter kKimiAdapter;
  static const LlamaAdapter kLlamaAdapter;
  static const MiniMaxMoEAdapter kMiniMaxAdapter;
  static const TernaryAdapter kTernaryAdapter;
  static const std::array<const IUS4V6Adapter *, 9> kAdapters = {
      &kQwenAdapter,    &kGemmaAdapter,   &kBitNetAdapter,
      &kTernaryAdapter, &kLlamaAdapter,   &kDeepSeekAdapter,
      &kKimiAdapter,    &kMiniMaxAdapter, &kGlmAdapter};
  return kAdapters;
}

} // namespace

const IUS4V6Adapter *FindAdapterByModel(const std::string_view modelName) {
  const std::string normalized = Normalize(modelName);

  for (const IUS4V6Adapter *adapter : AdapterTable()) {
    if (normalized == Normalize(adapter->ModelName())) {
      return adapter;
    }
  }

  for (const IUS4V6Adapter *adapter : AdapterTable()) {
    if (normalized == Normalize(adapter->Family())) {
      return adapter;
    }
  }

  const IUS4V6Adapter *bestMatch = nullptr;
  std::size_t bestLength = 0;
  for (const IUS4V6Adapter *adapter : AdapterTable()) {
    const std::string family = Normalize(adapter->Family());
    if (normalized.find(family) != std::string::npos &&
        family.size() > bestLength) {
      bestMatch = adapter;
      bestLength = family.size();
    }
  }
  return bestMatch;
}

std::vector<const IUS4V6Adapter *> ListAdapters() {
  const auto &table = AdapterTable();
  return {table.begin(), table.end()};
}

} // namespace us4
