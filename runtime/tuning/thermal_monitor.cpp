#include "tuning/thermal_monitor.h"

namespace us4 {

ThermalDowngradeDecision DecideThermalDowngrade(const ThermalReading &reading) {
  ThermalDowngradeDecision decision;
  switch (reading.state) {
  case ThermalState::kNominal:
    decision.requiresDowngrade = false;
    decision.reason = "nominal";
    break;
  case ThermalState::kFair:
    decision.requiresDowngrade = false;
    decision.reason = "fair";
    break;
  case ThermalState::kSerious:
    decision.requiresDowngrade = true;
    decision.reason = "serious-thermal";
    break;
  case ThermalState::kCritical:
    decision.requiresDowngrade = true;
    decision.reason = "critical-thermal";
    break;
  }
  return decision;
}

} // namespace us4
