#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace us4 {

enum class RuntimeMode {
  kFull,
  kBalancedPlus,
  kDegraded,
  kUltraLow,
  kMicro,
  kMicroPlus,
  kNano,
};

std::string_view ToString(RuntimeMode mode);
std::optional<RuntimeMode> ParseRuntimeMode(std::string_view value);
RuntimeMode SelectRuntimeModeFromMemoryGiB(unsigned long long memory_gib);
int RuntimeModeRank(RuntimeMode mode);
RuntimeMode MaxRuntimeMode(RuntimeMode lhs, RuntimeMode rhs);

}  // namespace us4
