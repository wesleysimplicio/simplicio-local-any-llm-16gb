#pragma once

#include <string>

#include "core/runtime_mode.h"

namespace us4 {

struct HardwareProbeResult {
  std::string platform;
  std::string architecture;
  std::string chip;
  unsigned long long unifiedMemoryGiB = 0;
  bool isAppleSilicon = false;
  bool hasMlx = false;
  bool hasMetal = false;
  bool hasNeon = false;
  bool hasAne = false;
  bool supportsCoreMl = false;
  unsigned int neonVectorBits = 0;
  bool hasPerformanceCores = false;
  bool hasEfficiencyCores = false;
  RuntimeMode recommendedMode = RuntimeMode::kNano;
};

class HardwareProbe {
public:
  static HardwareProbeResult Detect();
};

} // namespace us4
