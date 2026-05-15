#include "core/backend_selector.h"

#include <cctype>
#include <string>

#include "core/ius4v6_adapter.h"

namespace us4 {

namespace {

std::string Normalize(const std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (const char ch : value) {
    if (ch == '-' || ch == ' ') {
      normalized.push_back('_');
    } else {
      normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
  }
  return normalized;
}

bool SupportsBackend(const HardwareProbeResult& hardware,
                     const RuntimeMode mode,
                     const IUS4V6Adapter& adapter,
                     const BackendType backend) {
  switch (backend) {
    case BackendType::kScalarCpu:
      return true;
    case BackendType::kNeon:
      return hardware.hasNeon && mode != RuntimeMode::kFull;
    case BackendType::kMlx:
      return hardware.hasMlx && adapter.SupportsMlxBackend() &&
             mode != RuntimeMode::kMicro && mode != RuntimeMode::kMicroPlus && mode != RuntimeMode::kNano;
    case BackendType::kMetal:
      return hardware.hasMetal && adapter.SupportsMetalBackend() &&
             mode != RuntimeMode::kMicro && mode != RuntimeMode::kMicroPlus && mode != RuntimeMode::kNano;
    case BackendType::kAne:
      return hardware.hasAne && adapter.SupportsAneBackend() && mode == RuntimeMode::kFull;
  }
  return false;
}

BackendType PreferredBackend(const HardwareProbeResult& hardware,
                             const RuntimeMode mode,
                             const IUS4V6Adapter& adapter) {
  if (SupportsBackend(hardware, mode, adapter, BackendType::kMetal)) {
    return BackendType::kMetal;
  }
  if (SupportsBackend(hardware, mode, adapter, BackendType::kMlx)) {
    return BackendType::kMlx;
  }
  if (SupportsBackend(hardware, mode, adapter, BackendType::kNeon)) {
    return BackendType::kNeon;
  }
  return BackendType::kScalarCpu;
}

}  // namespace

std::string_view ToString(const BackendType backend) {
  switch (backend) {
    case BackendType::kScalarCpu:
      return "scalar";
    case BackendType::kNeon:
      return "neon";
    case BackendType::kMlx:
      return "mlx";
    case BackendType::kMetal:
      return "metal";
    case BackendType::kAne:
      return "ane";
  }
  return "scalar";
}

std::optional<BackendType> ParseBackendType(const std::string_view value) {
  const std::string normalized = Normalize(value);
  if (normalized == "scalar" || normalized == "cpu") {
    return BackendType::kScalarCpu;
  }
  if (normalized == "neon") {
    return BackendType::kNeon;
  }
  if (normalized == "mlx") {
    return BackendType::kMlx;
  }
  if (normalized == "metal") {
    return BackendType::kMetal;
  }
  if (normalized == "ane") {
    return BackendType::kAne;
  }
  return std::nullopt;
}

BackendSelection SelectBackend(const HardwareProbeResult& hardware,
                               const RuntimeMode mode,
                               const IUS4V6Adapter& adapter,
                               const std::optional<BackendType> requested) {
  BackendSelection selection;
  if (requested.has_value()) {
    if (SupportsBackend(hardware, mode, adapter, *requested)) {
      selection.selected = *requested;
      selection.reason = "requested";
      return selection;
    }
    selection.selected = PreferredBackend(hardware, mode, adapter);
    selection.fellBack = true;
    selection.reason = "requested-backend-unavailable";
    return selection;
  }

  selection.selected = PreferredBackend(hardware, mode, adapter);
  selection.reason = "auto";
  return selection;
}

}  // namespace us4
