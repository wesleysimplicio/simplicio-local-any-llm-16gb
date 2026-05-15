#pragma once

#include "adapters/dense_adapter_base.h"

namespace us4 {

class KimiMoEAdapter final : public DenseAdapterBase {
 public:
  KimiMoEAdapter();

  ArchitectureType Architecture() const override;
  bool SupportsMoe() const override;
  bool SupportsMlxBackend() const override;
  bool SupportsMetalBackend() const override;
  GenerationResult Generate(const GenerationRequest& request, const RuntimeContext& context) const override;

 protected:
  std::uint32_t Seed() const override;
  std::vector<std::string> Vocabulary() const override;
  std::string DefaultPromptToken() const override;
};

}  // namespace us4
