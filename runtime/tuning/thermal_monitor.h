#pragma once

#include <cstddef>
#include <string>

namespace us4 {

// Thermal monitor contract. Sprint 11 reads thermal signals from
// `IOPMrootDomain`/`powermetrics` on Apple Silicon. The runtime turns the
// reading into a downgrade decision that downstream dispatch consumes.

enum class ThermalState {
  kNominal,
  kFair,
  kSerious,
  kCritical,
};

struct ThermalReading {
  ThermalState state = ThermalState::kNominal;
  float celsius = 0.0F;
  std::string source = "synthetic";
};

struct ThermalDowngradeDecision {
  bool requiresDowngrade = false;
  std::string reason;
};

ThermalDowngradeDecision DecideThermalDowngrade(const ThermalReading& reading);

}  // namespace us4
