#include "speculative/speculative_telemetry.h"

#include <algorithm>

namespace us4 {

SpeculativeTelemetry ComputeSpeculativeTelemetry(const std::size_t draftAttempts,
                                                 const std::size_t acceptedTokens,
                                                 const std::size_t verifyWindow) {
  SpeculativeTelemetry telemetry;
  telemetry.draftAttempts = draftAttempts;
  telemetry.acceptedTokens = acceptedTokens;
  telemetry.rejectedTokens = draftAttempts > acceptedTokens
                                 ? draftAttempts - acceptedTokens
                                 : 0;
  telemetry.verifyWindow = verifyWindow;
  if (draftAttempts > 0) {
    telemetry.acceptanceRate =
        static_cast<float>(acceptedTokens) / static_cast<float>(draftAttempts);
  }
  return telemetry;
}

AdaptiveSpeculativePlan
PlanAdaptiveSpeculation(const AdaptiveSpeculativeState &state,
                        const AdaptiveSpeculativeConfig &config) {
  AdaptiveSpeculativePlan plan;
  const std::size_t minLookahead =
      std::max<std::size_t>(1U, config.minLookaheadTokens);
  const std::size_t maxLookahead =
      std::max(minLookahead, config.maxLookaheadTokens);
  plan.warmupActive = state.observedDrafts < config.warmupDrafts;

  if (state.attemptedTokens > 0U) {
    plan.observedAcceptanceRate =
        static_cast<float>(state.acceptedTokens) /
        static_cast<float>(state.attemptedTokens);
  }
  if (state.expertCacheLookups > 0U) {
    plan.observedExpertCacheHitRate =
        static_cast<float>(state.expertCacheHits) /
        static_cast<float>(state.expertCacheLookups);
  }
  const bool expertCacheWarm =
      state.expertCacheLookups == 0U ||
      plan.observedExpertCacheHitRate >=
          config.minExpertCacheHitRateForMtp;

  if (plan.warmupActive) {
    plan.lookaheadTokens = minLookahead;
  } else if (plan.observedAcceptanceRate <
             config.minAcceptanceRateForMtp) {
    plan.lookaheadTokens = minLookahead;
  } else if (plan.observedAcceptanceRate >=
             config.minAcceptanceRateForExpansion) {
    plan.lookaheadTokens =
        std::min<std::size_t>(maxLookahead,
                              std::max(minLookahead + 1U,
                                       state.lookaheadTokens + 1U));
  } else {
    plan.lookaheadTokens =
        std::min<std::size_t>(maxLookahead,
                              std::max(minLookahead, state.lookaheadTokens));
  }

  plan.verifyWindow = plan.lookaheadTokens;
  plan.mtpEnabled = !plan.warmupActive &&
                    expertCacheWarm &&
                    plan.observedAcceptanceRate >=
                        config.minAcceptanceRateForMtp &&
                    plan.lookaheadTokens > minLookahead;
  return plan;
}

void UpdateAdaptiveSpeculativeState(AdaptiveSpeculativeState &state,
                                    const SpeculativeTelemetry &telemetry,
                                    const AdaptiveSpeculativeConfig &config) {
  state.observedDrafts += 1U;
  state.acceptedTokens += telemetry.acceptedTokens;
  state.attemptedTokens += telemetry.draftAttempts;
  const AdaptiveSpeculativePlan nextPlan =
      PlanAdaptiveSpeculation(state, config);
  state.lookaheadTokens = nextPlan.lookaheadTokens;
}

void RecordExpertCacheLookup(AdaptiveSpeculativeState &state, const bool hit) {
  ++state.expertCacheLookups;
  if (hit) {
    ++state.expertCacheHits;
  }
}

AdaptiveSpeculativeConfig
Make16GbAdaptiveSpeculativeConfig(const std::size_t maxLookaheadTokens) {
  AdaptiveSpeculativeConfig config;
  config.warmupDrafts = 3U;
  config.minLookaheadTokens = 1U;
  config.maxLookaheadTokens =
      std::max<std::size_t>(1U, std::min<std::size_t>(2U, maxLookaheadTokens));
  config.minAcceptanceRateForMtp = 0.75F;
  config.minAcceptanceRateForExpansion = 0.95F;
  config.minExpertCacheHitRateForMtp = 0.60F;
  return config;
}

} // namespace us4
