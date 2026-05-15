#pragma once

#include "adapters/dense_adapter_base.h"

namespace us4 {

class GemmaAdapter final : public DenseAdapterBase {
 public:
  GemmaAdapter();

 protected:
  std::uint32_t Seed() const override;
  std::vector<std::string> Vocabulary() const override;
  std::string DefaultPromptToken() const override;
};

}  // namespace us4
