#pragma once

#include <cstddef>
#include <string_view>

#include "adapters/dense_adapter_base.h"

namespace us4 {

class BitNetAdapter final : public DenseAdapterBase {
 public:
  BitNetAdapter();

  ArchitectureType Architecture() const override;
  RuntimeMode MinimumMode() const override;

  // Sprint 05: packed weight format declared by the BitNet family. The
  // loader contract uses this string to route bitnet GGUF assets and
  // telemetry surfaces it so MICRO mode can attribute the route.
  static constexpr std::string_view kWeightFormat = "bitnet-b1.58";
  static constexpr std::size_t kPackedTernitsPerByte = 5U;

 protected:
  std::uint32_t Seed() const override;
  std::vector<std::string> Vocabulary() const override;
  std::string DefaultPromptToken() const override;
};

}  // namespace us4
