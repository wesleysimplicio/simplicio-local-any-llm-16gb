#pragma once

#include <cstddef>

namespace us4 {

// Speculative decoding telemetry. The struct is intentionally small so the
// scheduler can fill it on every batch step without overhead. All fields are
// CLI/bench-visible.

struct SpeculativeTelemetry {
  std::size_t draftAttempts = 0;
  std::size_t acceptedTokens = 0;
  std::size_t rejectedTokens = 0;
  std::size_t verifyWindow = 0;
  float acceptanceRate = 0.0F;
};

struct AdaptiveSpeculativeConfig {
  std::size_t warmupDrafts = 2;
  std::size_t minLookaheadTokens = 1;
  std::size_t maxLookaheadTokens = 4;
  float minAcceptanceRateForMtp = 0.5F;
  float minAcceptanceRateForExpansion = 0.85F;
  float minExpertCacheHitRateForMtp = 0.0F;
};

struct AdaptiveSpeculativeState {
  std::size_t observedDrafts = 0;
  std::size_t acceptedTokens = 0;
  std::size_t attemptedTokens = 0;
  std::size_t lookaheadTokens = 1;
  std::size_t expertCacheHits = 0;
  std::size_t expertCacheLookups = 0;
};

struct AdaptiveSpeculativePlan {
  std::size_t lookaheadTokens = 1;
  std::size_t verifyWindow = 1;
  bool mtpEnabled = false;
  bool warmupActive = true;
  float observedAcceptanceRate = 0.0F;
  float observedExpertCacheHitRate = 0.0F;
};

SpeculativeTelemetry ComputeSpeculativeTelemetry(std::size_t draftAttempts,
                                                 std::size_t acceptedTokens,
                                                 std::size_t verifyWindow);
AdaptiveSpeculativePlan
PlanAdaptiveSpeculation(const AdaptiveSpeculativeState &state,
                        const AdaptiveSpeculativeConfig &config);
void UpdateAdaptiveSpeculativeState(AdaptiveSpeculativeState &state,
                                    const SpeculativeTelemetry &telemetry,
                                    const AdaptiveSpeculativeConfig &config);
void RecordExpertCacheLookup(AdaptiveSpeculativeState &state, bool hit);
AdaptiveSpeculativeConfig Make16GbAdaptiveSpeculativeConfig(
    std::size_t maxLookaheadTokens);

}  // namespace us4
