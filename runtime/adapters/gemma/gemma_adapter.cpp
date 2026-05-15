#include "adapters/gemma/gemma_adapter.h"

namespace us4 {

GemmaAdapter::GemmaAdapter() : DenseAdapterBase("gemma", "gemma-2b-it") {}

std::uint32_t GemmaAdapter::Seed() const { return 22073U; }

std::vector<std::string> GemmaAdapter::Vocabulary() const {
  return {"gemma", "answers", "with", "cpu", "scalar", "reference", "path", "for",
          "local", "testing", "hello", "apple", "runtime", ".", "clean", "output"};
}

std::string GemmaAdapter::DefaultPromptToken() const { return "hello"; }

}  // namespace us4
