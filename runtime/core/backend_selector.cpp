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
      normalized.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
  }
  return normalized;
}

bool SupportsBackend(const HardwareProbeResult &hardware,
                     const RuntimeMode mode, const IUS4V6Adapter &adapter,
                     const BackendType backend) {
  switch (backend) {
  case BackendType::kScalarCpu:
    return true;
  case BackendType::kNeon:
    if (!hardware.hasNeon || hardware.neonVectorBits < 128U) {
      return false;
    }
    if (mode == RuntimeMode::kDegraded || mode == RuntimeMode::kUltraLow) {
      return hardware.hasPerformanceCores;
    }
    return (mode == RuntimeMode::kMicro || mode == RuntimeMode::kMicroPlus ||
            mode == RuntimeMode::kNano) &&
           (hardware.hasPerformanceCores || hardware.hasEfficiencyCores);
  case BackendType::kMlx:
    return hardware.hasMlx && adapter.SupportsMlxBackend() &&
           (mode == RuntimeMode::kFull || mode == RuntimeMode::kBalancedPlus ||
            mode == RuntimeMode::kDegraded || mode == RuntimeMode::kUltraLow);
  case BackendType::kMetal:
    return hardware.hasMetal && adapter.SupportsMetalBackend() &&
           (mode == RuntimeMode::kFull || mode == RuntimeMode::kBalancedPlus);
  case BackendType::kAne:
    return hardware.hasAne && adapter.SupportsAneBackend() &&
           mode == RuntimeMode::kFull;
  }
  return false;
}

BackendType PreferredBackend(const HardwareProbeResult &hardware,
                             const RuntimeMode mode,
                             const IUS4V6Adapter &adapter) {
  if (SupportsBackend(hardware, mode, adapter, BackendType::kAne)) {
    return BackendType::kAne;
  }
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

std::string_view AutoReasonFor(const BackendType backend) {
  switch (backend) {
  case BackendType::kMetal:
    return "auto-metal";
  case BackendType::kMlx:
    return "auto-mlx";
  case BackendType::kNeon:
    return "auto-neon";
  case BackendType::kScalarCpu:
    return "auto-scalar";
  case BackendType::kAne:
    return "auto-ane";
  }
  return "auto-scalar";
}

} // namespace

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

BackendSelection SelectBackend(const HardwareProbeResult &hardware,
                               const RuntimeMode mode,
                               const IUS4V6Adapter &adapter,
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
  selection.reason = AutoReasonFor(selection.selected);
  return selection;
}

} // namespace us4
