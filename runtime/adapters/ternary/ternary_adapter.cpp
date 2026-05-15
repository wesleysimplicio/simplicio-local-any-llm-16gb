#include "adapters/ternary/ternary_adapter.h"

namespace us4 {

TernaryAdapter::TernaryAdapter() : DenseAdapterBase("ternary", "pt-bitnet-ternary-2b") {}

ArchitectureType TernaryAdapter::Architecture() const { return ArchitectureType::kTernary; }

RuntimeMode TernaryAdapter::MinimumMode() const { return RuntimeMode::kMicro; }

std::uint32_t TernaryAdapter::Seed() const { return 30021U; }

std::vector<std::string> TernaryAdapter::Vocabulary() const {
  return {"ternary", "lookup", "table", "micro", "local", "inference", "apple", "runtime",
          "responds", "with", "small", "footprint", ".", "cheap", "fast", "hi"};
}

std::string TernaryAdapter::DefaultPromptToken() const { return "hi"; }

}  // namespace us4
