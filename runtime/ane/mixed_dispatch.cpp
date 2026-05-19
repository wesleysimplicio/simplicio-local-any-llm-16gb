#include "ane/mixed_dispatch.h"

namespace us4 {

MixedDispatchPlan BuildMixedDispatchPlan(const LayerOffloadPlan &offload) {
  MixedDispatchPlan plan;
  for (const auto &layer : offload.aneLayers) {
    plan.steps.push_back({layer, "ane"});
    ++plan.aneSteps;
  }
  for (const auto &layer : offload.metalLayers) {
    plan.steps.push_back({layer, "metal"});
    ++plan.metalSteps;
  }
  return plan;
}

} // namespace us4
