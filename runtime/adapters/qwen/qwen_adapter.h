#pragma once

#include "adapters/dense_adapter_base.h"

namespace us4 {

class QwenAdapter final : public DenseAdapterBase {
 public:
  QwenAdapter();

 protected:
  std::uint32_t Seed() const override;
  std::vector<std::string> Vocabulary() const override;
  std::string DefaultPromptToken() const override;
};

}  // namespace us4
