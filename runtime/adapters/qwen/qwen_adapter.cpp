#include "adapters/qwen/qwen_adapter.h"

namespace us4 {

QwenAdapter::QwenAdapter() : DenseAdapterBase("qwen", "qwen-0.5b") {}

std::uint32_t QwenAdapter::Seed() const { return 41051U; }

std::vector<std::string> QwenAdapter::Vocabulary() const {
  return {"hello", "from", "us4", "apple", "runtime", "cpu", "baseline", "ready",
          "local", "model", "qwen", "says", "hi", ".", "fast", "stable"};
}

std::string QwenAdapter::DefaultPromptToken() const { return "hi"; }

}  // namespace us4
