#include <cstdlib>
#include <stop_token>
#include <string>
#include <vector>

#include "moe/expert_pager.h"
#include "speculative/lossless_speculative_session.h"
#include "speculative/speculative_telemetry.h"

namespace {

bool Expect(const bool condition) { return condition; }

} // namespace

int main() {
  bool ok = true;

  us4::ExpertPager pager(2U, 2U, 1U);
  pager.Touch("layer-0/expert-a");
  pager.Touch("layer-0/expert-a");
  pager.Touch("layer-1/expert-b");
  pager.Touch("layer-1/expert-b");
  pager.Touch("layer-1/expert-b");
  const us4::ExpertPagerSnapshot cache = pager.Snapshot();
  ok &= Expect(pager.IsPinned("layer-1/expert-b"));
  ok &= Expect(!pager.IsPinned("layer-0/expert-a"));
  ok &= Expect(cache.expertsCoveringHalfOfTouches == 1U);
  ok &= Expect(cache.pinDemotionCount == 1U);
  ok &= Expect(pager.ExportUsageJson().find("\"hits\":") != std::string::npos);

  us4::AdaptiveSpeculativeState adaptive;
  const us4::AdaptiveSpeculativeConfig config =
      us4::Make16GbAdaptiveSpeculativeConfig(4U);
  for (std::size_t index = 0U; index < config.warmupDrafts; ++index) {
    us4::UpdateAdaptiveSpeculativeState(
        adaptive, us4::ComputeSpeculativeTelemetry(2U, 2U, 1U), config);
  }
  for (int index = 0; index < 3; ++index) {
    us4::RecordExpertCacheLookup(adaptive, true);
  }
  ok &= Expect(us4::PlanAdaptiveSpeculation(adaptive, config).mtpEnabled);

  us4::LosslessSpeculativeSession session(
      {.maxDraftTokens = 4U, .maxRounds = 1U, .maxCommittedTokens = 3U});
  const us4::LosslessSpeculativeRound mismatch =
      session.RunRound({10, 11, 42, 13}, {10, 11, 12, 13});
  ok &= Expect(mismatch.committedTokens == std::vector<int>({10, 11, 42}));
  ok &= Expect(mismatch.matchesAuthoritativePath);
  ok &= Expect(session.Metrics().acceptedTokens == 2U);
  ok &= Expect(session.Metrics().rejectedTokens == 2U);

  std::stop_source stop;
  stop.request_stop();
  us4::LosslessSpeculativeSession cancelled;
  const us4::LosslessSpeculativeRound cancelledRound =
      cancelled.RunRound({1}, {9}, stop.get_token());
  ok &= Expect(cancelledRound.committedTokens.empty());
  ok &= Expect(cancelledRound.stopReason ==
               us4::SpeculativeStopReason::kCancelled);

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
