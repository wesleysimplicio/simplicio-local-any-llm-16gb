#include <gtest/gtest.h>

#include "moe/expert_pager.h"
#include "moe/router.h"

TEST(MoeContractTest, RouterReturnsScoresSortedByTopK) {
  us4::Router router;
  const auto topk = router.TopK({0.1F, 0.8F, 0.4F, 0.7F}, 3);

  ASSERT_EQ(topk.size(), 3U);
  EXPECT_EQ(topk[0].expert, 1U);
  EXPECT_FLOAT_EQ(topk[0].logit, 0.8F);
  EXPECT_EQ(topk[1].expert, 3U);
  EXPECT_FLOAT_EQ(topk[1].logit, 0.7F);
  EXPECT_EQ(topk[2].expert, 2U);
  EXPECT_FLOAT_EQ(topk[2].logit, 0.4F);
  EXPECT_GT(topk[0].score, topk[1].score);
  EXPECT_GT(topk[1].score, topk[2].score);
}

TEST(MoeContractTest, RouterClampsKToAvailableExperts) {
  us4::Router router;
  const auto topk = router.TopK({0.5F, 0.2F}, 4);

  ASSERT_EQ(topk.size(), 2U);
  EXPECT_EQ(topk[0].expert, 0U);
  EXPECT_EQ(topk[1].expert, 1U);
}

TEST(MoeContractTest, RouterDecisionExposesEntropyMassAndBalance) {
  us4::Router router;
  const auto decision = router.RouteTopK({2.0F, 1.0F, 0.0F, -1.0F}, 2);

  ASSERT_EQ(decision.selected.size(), 2U);
  EXPECT_EQ(decision.totalExperts, 4U);
  EXPECT_GT(decision.entropy, 0.0F);
  EXPECT_GT(decision.loadBalance, 0.0F);
  EXPECT_LE(decision.loadBalance, 1.0F);
  EXPECT_GT(decision.selectedMass, 0.0F);
  EXPECT_LE(decision.selectedMass, 1.0F);
  EXPECT_FLOAT_EQ(decision.selected[0].logit, 2.0F);
  ASSERT_TRUE(router.LastDecision().has_value());
  EXPECT_EQ(router.LastDecision()->selected.size(), 2U);
  EXPECT_FLOAT_EQ(router.LastDecision()->selectedMass, decision.selectedMass);
}

TEST(MoeContractTest, ExpertPagerRetainsMostFrequentlyTouchedExperts) {
  us4::ExpertPager pager(2);
  pager.Touch("expert-a");
  pager.Touch("expert-b");
  pager.Touch("expert-a");
  pager.Touch("expert-c");

  EXPECT_EQ(pager.ResidentCount(), 2U);
  EXPECT_TRUE(pager.IsResident("expert-a"));
  EXPECT_TRUE(pager.IsResident("expert-c"));
  EXPECT_FALSE(pager.IsResident("expert-b"));
  EXPECT_EQ(pager.LoadCount(), 3U);
  EXPECT_EQ(pager.ReuseCount(), 1U);
  EXPECT_EQ(pager.EvictionCount(), 1U);
}

TEST(MoeContractTest, ExpertPagerIgnoresUnknownResidencyQueries) {
  us4::ExpertPager pager(1);
  pager.Touch("expert-a");

  EXPECT_FALSE(pager.IsResident("expert-missing"));
}

TEST(MoeContractTest, ExpertPagerSnapshotKeepsVisibleResidentState) {
  us4::ExpertPager pager(2);
  pager.Touch("expert-a");
  pager.Touch("expert-b");
  pager.Touch("expert-a");

  const us4::ExpertPagerSnapshot snapshot = pager.Snapshot();

  EXPECT_EQ(snapshot.residentCount, 2U);
  EXPECT_EQ(snapshot.loadCount, 2U);
  EXPECT_EQ(snapshot.reuseCount, 1U);
  EXPECT_EQ(snapshot.evictionCount, 0U);
  ASSERT_EQ(snapshot.residents.size(), 2U);
  EXPECT_EQ(snapshot.residents[0], "expert-a");
  EXPECT_EQ(snapshot.residents[1], "expert-b");
}
