#include "tuning/auto_tuner.h"

#include <algorithm>

namespace us4 {

AutoTunerProfile
SelectAutoTunerProfile(const HardwareProbeResult &hardware,
                       const std::vector<AutoTunerCandidate> &candidates) {
  AutoTunerProfile profile;
  profile.chip = hardware.chip;
  if (candidates.empty()) {
    return profile;
  }
  auto best = std::min_element(
      candidates.begin(), candidates.end(),
      [](const AutoTunerCandidate &lhs, const AutoTunerCandidate &rhs) {
        if (lhs.observedLatencyMs != rhs.observedLatencyMs) {
          return lhs.observedLatencyMs < rhs.observedLatencyMs;
        }
        if (lhs.batchSize != rhs.batchSize) {
          return lhs.batchSize > rhs.batchSize;
        }
        if (lhs.tileRows != rhs.tileRows) {
          return lhs.tileRows > rhs.tileRows;
        }
        return lhs.tileCols > rhs.tileCols;
      });
  profile.tileRows = best->tileRows;
  profile.tileCols = best->tileCols;
  profile.batchSize = best->batchSize;
  profile.estimatedLatencyMs = best->observedLatencyMs;
  return profile;
}

} // namespace us4
