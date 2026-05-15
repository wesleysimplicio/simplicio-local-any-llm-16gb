#pragma once

#include "adapters/dense_adapter_base.h"

namespace us4 {

class BitNetAdapter final : public DenseAdapterBase {
 public:
  BitNetAdapter();

  ArchitectureType Architecture() const override;
  RuntimeMode MinimumMode() const override;

 protected:
  std::uint32_t Seed() const override;
  std::vector<std::string> Vocabulary() const override;
  std::string DefaultPromptToken() const override;
};

}  // namespace us4
