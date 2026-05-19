#include "tuning/thermal_monitor.h"

namespace us4 {

namespace {

RuntimeMode ClampRuntimeMode(const RuntimeMode requestedMode,
                             const RuntimeMode capMode) {
  return RuntimeModeRank(requestedMode) <= RuntimeModeRank(capMode)
             ? requestedMode
             : capMode;
}

} // namespace

std::string_view ToString(const ThermalPressureLevel level) {
  switch (level) {
  case ThermalPressureLevel::kUnavailable:
    return "unavailable";
  case ThermalPressureLevel::kNominal:
    return "nominal";
  case ThermalPressureLevel::kElevated:
    return "elevated";
  case ThermalPressureLevel::kCritical:
    return "critical";
  }
  return "unavailable";
}

ThermalMonitor::ThermalMonitor(const HardwareProbeResult &hardware) {
  available_ = hardware.isAppleSilicon || hardware.hasMetal || hardware.hasAne;
  sample_.available = available_;
  sample_.source = available_ ? "probe-derived" : "none";

  if (!available_) {
    sample_.level = ThermalPressureLevel::kUnavailable;
    sample_.reason = "thermal-unavailable";
    lastDecision_.reason = sample_.reason;
    return;
  }

  if (!hardware.hasPerformanceCores || hardware.unifiedMemoryGiB <= 12ULL) {
    sample_.level = ThermalPressureLevel::kCritical;
    sample_.reason = "thermal-critical-cap";
  } else if (hardware.hasAne && hardware.supportsCoreMl &&
             hardware.unifiedMemoryGiB <= 24ULL) {
    sample_.level = ThermalPressureLevel::kElevated;
    sample_.reason = "thermal-elevated-cap";
  } else {
    sample_.level = ThermalPressureLevel::kNominal;
    sample_.reason = "thermal-nominal";
  }

  lastDecision_.level = sample_.level;
  lastDecision_.reason = sample_.reason;
}

bool ThermalMonitor::Available() const { return available_; }

std::string_view ThermalMonitor::Reason() const { return sample_.reason; }

ThermalSample ThermalMonitor::Sample() const { return sample_; }

ThermalDecision ThermalMonitor::Decide(const RuntimeMode requestedMode) {
  lastDecision_.requestedMode = requestedMode;
  lastDecision_.level = sample_.level;
  lastDecision_.downgraded = false;
  lastDecision_.reason = sample_.reason;

  if (!available_) {
    lastDecision_.effectiveMode = requestedMode;
    return lastDecision_;
  }

  switch (sample_.level) {
  case ThermalPressureLevel::kNominal:
    lastDecision_.effectiveMode = requestedMode;
    break;
  case ThermalPressureLevel::kElevated:
    lastDecision_.effectiveMode =
        ClampRuntimeMode(requestedMode, RuntimeMode::kBalancedPlus);
    lastDecision_.downgraded =
        lastDecision_.effectiveMode != lastDecision_.requestedMode;
    break;
  case ThermalPressureLevel::kCritical:
    lastDecision_.effectiveMode =
        ClampRuntimeMode(requestedMode, RuntimeMode::kMicroPlus);
    lastDecision_.downgraded =
        lastDecision_.effectiveMode != lastDecision_.requestedMode;
    break;
  case ThermalPressureLevel::kUnavailable:
    lastDecision_.effectiveMode = requestedMode;
    break;
  }

  return lastDecision_;
}

const ThermalDecision &ThermalMonitor::LastDecision() const {
  return lastDecision_;
}

} // namespace us4
