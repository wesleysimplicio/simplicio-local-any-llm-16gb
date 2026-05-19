#pragma once

#include <cstddef>
#include <vector>

#include "moe/router.h"

namespace us4 {

// Sprint 08 routing telemetry.
// Entropy summarises how spread the router decision is over the available
// experts; lower entropy means a peakier distribution. LoadBalanceLoss
// measures how far the realized assignment deviates from a uniform load,
// which the runtime uses to detect router collapse.

float RouterEntropy(const std::vector<float> &logits);

float RouterLoadBalanceLoss(const std::vector<float> &logits, std::size_t k);

// Materialize the top-k decision together with its entropy/load-balance
// signals so the adapter result and benchmark evidence can attribute the
// routing behaviour.
struct RoutingTelemetry {
  std::vector<ExpertScore> topK;
  float entropy = 0.0F;
  float loadBalanceLoss = 0.0F;
  std::size_t consideredExperts = 0;
};

RoutingTelemetry ComputeRoutingTelemetry(const std::vector<float> &logits,
                                         std::size_t k);

} // namespace us4
