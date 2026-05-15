#pragma once

#include <optional>
#include <string_view>

#include "core/hardware_probe.h"

namespace us4 {

class IUS4V6Adapter;

enum class BackendType {
  kScalarCpu,
  kNeon,
  kMlx,
  kMetal,
  kAne,
};

struct BackendSelection {
  BackendType selected = BackendType::kScalarCpu;
  bool fellBack = false;
  std::string_view reason = "default";
};

std::string_view ToString(BackendType backend);
std::optional<BackendType> ParseBackendType(std::string_view value);
BackendSelection SelectBackend(const HardwareProbeResult& hardware,
                               RuntimeMode mode,
                               const IUS4V6Adapter& adapter,
                               std::optional<BackendType> requested = std::nullopt);

}  // namespace us4
