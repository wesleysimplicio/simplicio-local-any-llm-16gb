#pragma once

#include <string>
#include <string_view>

#include "core/hardware_probe.h"
#include "core/runtime_mode.h"

namespace us4 {

enum class ThermalPressureLevel {
  kUnavailable,
  kNominal,
  kElevated,
  kCritical,
};

struct ThermalSample {
  bool available = false;
  ThermalPressureLevel level = ThermalPressureLevel::kUnavailable;
  std::string source = "none";
  std::string reason = "thermal-unavailable";
};

struct ThermalDecision {
  RuntimeMode requestedMode = RuntimeMode::kNano;
  RuntimeMode effectiveMode = RuntimeMode::kNano;
  ThermalPressureLevel level = ThermalPressureLevel::kUnavailable;
  bool downgraded = false;
  std::string reason = "thermal-unavailable";
};

std::string_view ToString(ThermalPressureLevel level);

class ThermalMonitor {
public:
  ThermalMonitor() = default;
  explicit ThermalMonitor(const HardwareProbeResult &hardware);

  bool Available() const;
  std::string_view Reason() const;
  ThermalSample Sample() const;
  ThermalDecision Decide(RuntimeMode requestedMode);
  const ThermalDecision &LastDecision() const;

private:
  bool available_ = false;
  ThermalSample sample_;
  ThermalDecision lastDecision_;
};

} // namespace us4
