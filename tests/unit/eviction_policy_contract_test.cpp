#include <gtest/gtest.h>

#include "kv/eviction_policy.h"

TEST(EvictionPolicyContractTest, ChoosesLowestHitCountFirst) {
  const std::vector<us4::EvictionCandidate> candidates = {
      {"a", 5U, 100U},
      {"b", 1U, 200U},
      {"c", 3U, 50U},
  };

  const auto decision = us4::SelectEvictionVictim(candidates);

  EXPECT_TRUE(decision.valid);
  EXPECT_EQ(decision.key, "b");
  EXPECT_EQ(decision.reason, "lru-frequency");
}

TEST(EvictionPolicyContractTest, BreaksTiesByOldestLastTouch) {
  const std::vector<us4::EvictionCandidate> candidates = {
      {"old", 2U, 10U},
      {"new", 2U, 80U},
  };

  const auto decision = us4::SelectEvictionVictim(candidates);

  EXPECT_TRUE(decision.valid);
  EXPECT_EQ(decision.key, "old");
}

TEST(EvictionPolicyContractTest, ReportsColdReasonForUnusedPages) {
  const std::vector<us4::EvictionCandidate> candidates = {
      {"cold", 0U, 1U},
      {"warm", 4U, 2U},
  };

  const auto decision = us4::SelectEvictionVictim(candidates);

  EXPECT_TRUE(decision.valid);
  EXPECT_EQ(decision.key, "cold");
  EXPECT_EQ(decision.reason, "cold");
}

TEST(EvictionPolicyContractTest, ReturnsInvalidForEmptyCandidates) {
  const auto decision = us4::SelectEvictionVictim({});

  EXPECT_FALSE(decision.valid);
  EXPECT_TRUE(decision.key.empty());
}
