#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/hardware_probe.h"

namespace us4 {

// AutoTuner contract surface for Sprint 12.
//
// At runtime startup the tuner runs a small bench across a few candidate
// tile shapes and batch sizes, then picks the one with the lowest observed
// latency. The picked profile is deterministic for a given hardware
// snapshot and tunable through the profile cache.

struct AutoTunerCandidate {
  std::size_t tileRows = 0;
  std::size_t tileCols = 0;
  std::size_t batchSize = 0;
  float observedLatencyMs = 0.0F;
};

struct AutoTunerProfile {
  std::string chip;
  std::size_t tileRows = 0;
  std::size_t tileCols = 0;
  std::size_t batchSize = 0;
  float estimatedLatencyMs = 0.0F;
};

AutoTunerProfile
SelectAutoTunerProfile(const HardwareProbeResult& hardware,
                       const std::vector<AutoTunerCandidate>& candidates);

}  // namespace us4
