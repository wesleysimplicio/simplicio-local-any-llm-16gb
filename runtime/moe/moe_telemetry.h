#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace us4 {

// MoE telemetry container. The runtime fills this struct per generation step
// so adapters can expose routing + paging behaviour in the same observability
// envelope used by KV/backend telemetry.

struct ExpertResidencyEvent {
  std::string expertId;
  bool wasResident = false;
  bool becameResident = false;
  std::size_t hitCount = 0;
};

struct MoeTelemetrySnapshot {
  std::size_t routedExperts = 0;
  std::size_t residentExperts = 0;
  std::size_t evictedExperts = 0;
  float routerEntropy = 0.0F;
  float routerLoadBalanceLoss = 0.0F;
  float prefetchHitRatio = 0.0F;
  std::vector<ExpertResidencyEvent> events;
};

// Fold a routing decision plus a residency event list into the telemetry
// snapshot. The function never throws and stays cheap so it can run on every
// decode step.
MoeTelemetrySnapshot
BuildMoeTelemetrySnapshot(std::size_t routedExperts, std::size_t residentExperts,
                          std::size_t evictedExperts, float routerEntropy,
                          float routerLoadBalanceLoss, float prefetchHitRatio,
                          std::vector<ExpertResidencyEvent> events);

} // namespace us4
