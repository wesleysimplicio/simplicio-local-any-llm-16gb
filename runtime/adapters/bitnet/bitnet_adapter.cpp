#include "adapters/bitnet/bitnet_adapter.h"

namespace us4 {

BitNetAdapter::BitNetAdapter() : DenseAdapterBase("bitnet", "bitnet-b1.58-2b") {}

ArchitectureType BitNetAdapter::Architecture() const { return ArchitectureType::kTernary; }

RuntimeMode BitNetAdapter::MinimumMode() const { return RuntimeMode::kMicro; }

std::uint32_t BitNetAdapter::Seed() const { return 15802U; }

std::vector<std::string> BitNetAdapter::Vocabulary() const {
  return {"bitnet", "micro", "path", "local", "compact", "tokens", "apple", "runtime",
          "packed", "weights", "respond", "hi", ".", "tiny", "fast", "stable"};
}

std::string BitNetAdapter::DefaultPromptToken() const { return "hi"; }

}  // namespace us4
