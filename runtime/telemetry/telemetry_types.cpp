#include "telemetry/telemetry_types.h"

namespace us4::telemetry {

std::string_view toString(TelemetryCategory category) {
  switch (category) {
    case TelemetryCategory::kLatency:
      return "latency";
    case TelemetryCategory::kTokenThroughput:
      return "token_throughput";
    case TelemetryCategory::kMemory:
      return "memory";
    case TelemetryCategory::kModeTransition:
      return "mode_transition";
  }

  return "unknown";
}

std::string_view toString(MetricUnit unit) {
  switch (unit) {
    case MetricUnit::kMicroseconds:
      return "microseconds";
    case MetricUnit::kTokensPerSecond:
      return "tokens_per_second";
    case MetricUnit::kBytes:
      return "bytes";
    case MetricUnit::kCount:
      return "count";
  }

  return "unknown";
}

std::string_view toString(RuntimeMode mode) {
  switch (mode) {
    case RuntimeMode::kFull:
      return "FULL";
    case RuntimeMode::kBalancedPlus:
      return "BALANCED_PLUS";
    case RuntimeMode::kDegraded:
      return "DEGRADED";
    case RuntimeMode::kUltraLow:
      return "ULTRA_LOW";
    case RuntimeMode::kMicro:
      return "MICRO";
    case RuntimeMode::kMicroPlus:
      return "MICRO_PLUS";
    case RuntimeMode::kNano:
      return "NANO";
  }

  return "UNKNOWN";
}

std::string_view toString(ModeTransitionReason reason) {
  switch (reason) {
    case ModeTransitionReason::kHardwareProbe:
      return "hardware_probe";
    case ModeTransitionReason::kMemoryPressure:
      return "memory_pressure";
    case ModeTransitionReason::kCorrectnessFallback:
      return "correctness_fallback";
    case ModeTransitionReason::kThermalPressure:
      return "thermal_pressure";
    case ModeTransitionReason::kManualOverride:
      return "manual_override";
    case ModeTransitionReason::kUnknown:
      return "unknown";
  }

  return "unknown";
}

}  // namespace us4::telemetry
