#include <gtest/gtest.h>

#include "scheduler/continuous_batcher.h"

namespace {

TEST(SchedulerContractTest, PreservesSingleSessionPassthrough) {
  const us4::ContinuousBatcher batcher(6U);
  const us4::BatchDecision decision = batcher.Schedule({{"solo", 9U, 1U, 0U}});

  ASSERT_TRUE(decision.singleSessionPassthrough);
  ASSERT_EQ(decision.totalGrantedTokens, 6U);
  ASSERT_EQ(decision.activeSessions, 1U);
  ASSERT_EQ(decision.fairnessRounds, 1U);
  ASSERT_EQ(decision.slices.size(), 1U);
  EXPECT_EQ(decision.slices.front().sessionId, "solo");
  EXPECT_EQ(decision.slices.front().grantedTokens, 6U);
  EXPECT_EQ(decision.slices.front().roundsVisited, 1U);
}

TEST(SchedulerContractTest, AlternatesEqualSessionsFairly) {
  const us4::ContinuousBatcher batcher(4U);
  const us4::BatchDecision decision =
      batcher.Schedule({{"alpha", 3U, 1U, 0U}, {"beta", 3U, 1U, 1U}});

  ASSERT_FALSE(decision.singleSessionPassthrough);
  ASSERT_EQ(decision.totalGrantedTokens, 4U);
  ASSERT_EQ(decision.activeSessions, 2U);
  ASSERT_EQ(decision.fairnessRounds, 2U);
  ASSERT_EQ(decision.slices.size(), 2U);
  EXPECT_EQ(decision.slices[0].sessionId, "alpha");
  EXPECT_EQ(decision.slices[0].grantedTokens, 2U);
  EXPECT_EQ(decision.slices[1].sessionId, "beta");
  EXPECT_EQ(decision.slices[1].grantedTokens, 2U);
}

TEST(SchedulerContractTest, HonorsFairnessWeightWithoutStarvation) {
  const us4::ContinuousBatcher batcher(6U);
  const us4::BatchDecision decision =
      batcher.Schedule({{"heavy", 8U, 2U, 0U}, {"light", 8U, 1U, 1U}});

  ASSERT_EQ(decision.totalGrantedTokens, 6U);
  ASSERT_EQ(decision.fairnessRounds, 2U);
  ASSERT_EQ(decision.slices.size(), 2U);
  EXPECT_EQ(decision.slices[0].sessionId, "heavy");
  EXPECT_EQ(decision.slices[0].grantedTokens, 4U);
  EXPECT_EQ(decision.slices[1].sessionId, "light");
  EXPECT_EQ(decision.slices[1].grantedTokens, 2U);
  EXPECT_EQ(decision.slices[0].roundsVisited, 2U);
  EXPECT_EQ(decision.slices[1].roundsVisited, 2U);
}

} // namespace
