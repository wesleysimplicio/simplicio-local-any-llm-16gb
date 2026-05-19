#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ane/layer_offloader.h"

namespace us4 {

// MixedDispatchPlan combines an ANE layer plan with the rest of the dense
// pipeline. The plan is intentionally observable: each step records the
// chosen backend so the runtime can attribute fallback events.

struct MixedDispatchStep {
  std::string layerName;
  std::string backend; // "ane" or "metal"
};

struct MixedDispatchPlan {
  std::vector<MixedDispatchStep> steps;
  std::size_t aneSteps = 0;
  std::size_t metalSteps = 0;
};

MixedDispatchPlan BuildMixedDispatchPlan(const LayerOffloadPlan& offload);

}  // namespace us4
