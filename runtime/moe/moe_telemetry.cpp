#include "moe/moe_telemetry.h"

#include <utility>

namespace us4 {

MoeTelemetrySnapshot
BuildMoeTelemetrySnapshot(const std::size_t routedExperts,
                          const std::size_t residentExperts,
                          const std::size_t evictedExperts,
                          const float routerEntropy,
                          const float routerLoadBalanceLoss,
                          const float prefetchHitRatio,
                          std::vector<ExpertResidencyEvent> events) {
  MoeTelemetrySnapshot snapshot;
  snapshot.routedExperts = routedExperts;
  snapshot.residentExperts = residentExperts;
  snapshot.evictedExperts = evictedExperts;
  snapshot.routerEntropy = routerEntropy;
  snapshot.routerLoadBalanceLoss = routerLoadBalanceLoss;
  snapshot.prefetchHitRatio = prefetchHitRatio;
  snapshot.events = std::move(events);
  return snapshot;
}

} // namespace us4
