#include <gtest/gtest.h>

#include <stop_token>
#include <string>
#include <vector>

#include "moe/expert_pager.h"
#include "speculative/lossless_speculative_session.h"
#include "speculative/speculative_telemetry.h"

TEST(AdaptiveExpertCacheContractTest,
     RebalancesPinsToTheMeasuredHotSetAndExportsHistogram) {
  us4::ExpertPager pager(2U, 2U, 1U);
  pager.Touch("layer-0/expert-a");
  pager.Touch("layer-0/expert-a");
  EXPECT_TRUE(pager.IsPinned("layer-0/expert-a"));

  pager.Touch("layer-1/expert-b");
  pager.Touch("layer-1/expert-b");
  pager.Touch("layer-1/expert-b");

  const us4::ExpertPagerSnapshot snapshot = pager.Snapshot();
  EXPECT_FALSE(pager.IsPinned("layer-0/expert-a"));
  EXPECT_TRUE(pager.IsPinned("layer-1/expert-b"));
  EXPECT_EQ(snapshot.learnedPinCount, 1U);
  EXPECT_GE(snapshot.pinPromotionCount, 2U);
  EXPECT_GE(snapshot.pinDemotionCount, 1U);
  EXPECT_EQ(snapshot.lookupCount, 5U);
  EXPECT_GT(snapshot.hitRatio, 0.0);
  EXPECT_EQ(snapshot.expertsCoveringHalfOfTouches, 1U);
  EXPECT_NE(pager.ExportUsageJson().find("\"experts_covering_50_percent\":1"),
            std::string::npos);

  pager.Touch("layer-2/expert-\"quoted\"");
  EXPECT_NE(pager.ExportUsageJson().find("expert-\\\"quoted\\\""),
            std::string::npos);
}

TEST(AdaptiveExpertCacheContractTest, WarmupLoadsOnlyLearnedPinsWithinCapacity) {
  us4::ExpertPager pager(2U, 2U, 1U);
  pager.RestoreUsageHistogram(
      {{"expert-a", 9U, true}, {"expert-b", 3U, false},
       {"expert-c", 1U, false}});

  const std::vector<std::string> warmed = pager.WarmupLearnedPins();

  EXPECT_LE(pager.ResidentCount(), 2U);
  EXPECT_EQ(warmed, std::vector<std::string>{"expert-a"});
  EXPECT_TRUE(pager.IsResident("expert-a"));
}

TEST(AdaptiveSpeculative16GbContractTest,
     MtpWaitsForBothAcceptanceAndExpertCacheHitRate) {
  const us4::AdaptiveSpeculativeConfig config =
      us4::Make16GbAdaptiveSpeculativeConfig(4U);
  us4::AdaptiveSpeculativeState state;
  for (std::size_t draft = 0; draft < config.warmupDrafts; ++draft) {
    us4::UpdateAdaptiveSpeculativeState(
        state, us4::ComputeSpeculativeTelemetry(2U, 2U, 1U), config);
  }
  us4::RecordExpertCacheLookup(state, false);
  us4::RecordExpertCacheLookup(state, false);
  EXPECT_FALSE(us4::PlanAdaptiveSpeculation(state, config).mtpEnabled);

  for (int lookup = 0; lookup < 4; ++lookup) {
    us4::RecordExpertCacheLookup(state, true);
  }
  const us4::AdaptiveSpeculativePlan warm =
      us4::PlanAdaptiveSpeculation(state, config);
  EXPECT_TRUE(warm.mtpEnabled);
  EXPECT_LE(warm.lookaheadTokens, 2U);
  EXPECT_GE(warm.observedExpertCacheHitRate, 0.60F);
}

TEST(AdaptiveSpeculative16GbContractTest,
     DenseModelsWithoutExpertLookupsUseAcceptanceGateOnly) {
  const us4::AdaptiveSpeculativeConfig config =
      us4::Make16GbAdaptiveSpeculativeConfig(2U);
  us4::AdaptiveSpeculativeState state;
  for (std::size_t draft = 0; draft < config.warmupDrafts; ++draft) {
    us4::UpdateAdaptiveSpeculativeState(
        state, us4::ComputeSpeculativeTelemetry(2U, 2U, 1U), config);
  }

  EXPECT_EQ(state.expertCacheLookups, 0U);
  EXPECT_TRUE(us4::PlanAdaptiveSpeculation(state, config).mtpEnabled);
}

TEST(LosslessSpeculativeSessionContractTest,
     MismatchCommitsOnlyAuthoritativePrefixAndFallback) {
  us4::LosslessSpeculativeSession session(
      {.maxDraftTokens = 4U, .maxRounds = 2U, .maxCommittedTokens = 8U});
  const us4::LosslessSpeculativeRound round =
      session.RunRound({4, 5, 42, 8}, {4, 5, 6, 7});

  EXPECT_EQ(round.committedTokens, std::vector<int>({4, 5, 42}));
  EXPECT_TRUE(round.matchesAuthoritativePath);
  EXPECT_EQ(round.stopReason, us4::SpeculativeStopReason::kNone);
  EXPECT_EQ(session.Metrics().acceptedTokens, 2U);
  EXPECT_EQ(session.Metrics().rejectedTokens, 2U);
  EXPECT_DOUBLE_EQ(session.Metrics().acceptanceRate, 0.5);
}

TEST(LosslessSpeculativeSessionContractTest,
     CancellationAndLimitsNeverCommitUnverifiedTokens) {
  us4::LosslessSpeculativeSession session(
      {.maxDraftTokens = 3U, .maxRounds = 1U, .maxCommittedTokens = 2U});
  std::stop_source cancelled;
  cancelled.request_stop();
  const us4::LosslessSpeculativeRound cancelledRound =
      session.RunRound({1, 2, 3}, {9, 9, 9}, cancelled.get_token());
  EXPECT_TRUE(cancelledRound.committedTokens.empty());
  EXPECT_EQ(cancelledRound.stopReason,
            us4::SpeculativeStopReason::kCancelled);

  const us4::LosslessSpeculativeRound limited =
      session.RunRound({1, 2, 3}, {1, 2, 3});
  EXPECT_EQ(limited.committedTokens, std::vector<int>({1, 2}));
  EXPECT_TRUE(limited.matchesAuthoritativePath);
  EXPECT_EQ(limited.stopReason, us4::SpeculativeStopReason::kTokenLimit);

  const us4::LosslessSpeculativeRound exhausted =
      session.RunRound({4}, {4});
  EXPECT_TRUE(exhausted.committedTokens.empty());
  EXPECT_NE(exhausted.stopReason, us4::SpeculativeStopReason::kNone);
  EXPECT_EQ(session.Metrics().cancelledRounds, 1U);
}
