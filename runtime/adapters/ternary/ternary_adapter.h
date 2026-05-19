#pragma once

#include <cstddef>
#include <string_view>

#include "adapters/dense_adapter_base.h"

namespace us4 {

class TernaryAdapter final : public DenseAdapterBase {
 public:
  TernaryAdapter();

  ArchitectureType Architecture() const override;
  RuntimeMode MinimumMode() const override;

  // Sprint 05: PT-BitNet ternary weight format. The adapter declares the
  // explicit chunk size used by the lookup-table path so the loader and
  // telemetry stay aligned with the kernel contract.
  static constexpr std::string_view kWeightFormat = "pt-bitnet-ternary";
  static constexpr std::size_t kPackedTernitsPerByte = 4U;

 protected:
  std::uint32_t Seed() const override;
  std::vector<std::string> Vocabulary() const override;
  std::string DefaultPromptToken() const override;
};

}  // namespace us4
