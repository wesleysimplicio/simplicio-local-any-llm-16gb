#include <gtest/gtest.h>

#include "moe/expert_pager.h"
#include "moe/router.h"

TEST(MoeContractTest, RouterReturnsScoresSortedByTopK) {
  us4::Router router;
  const auto topk = router.TopK({0.1F, 0.8F, 0.4F, 0.7F}, 3);

  ASSERT_EQ(topk.size(), 3U);
  EXPECT_EQ(topk[0].expert, 1U);
  EXPECT_FLOAT_EQ(topk[0].score, 0.8F);
  EXPECT_EQ(topk[1].expert, 3U);
  EXPECT_FLOAT_EQ(topk[1].score, 0.7F);
  EXPECT_EQ(topk[2].expert, 2U);
  EXPECT_FLOAT_EQ(topk[2].score, 0.4F);
}

TEST(MoeContractTest, RouterClampsKToAvailableExperts) {
  us4::Router router;
  const auto topk = router.TopK({0.5F, 0.2F}, 4);

  ASSERT_EQ(topk.size(), 2U);
  EXPECT_EQ(topk[0].expert, 0U);
  EXPECT_EQ(topk[1].expert, 1U);
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
}

TEST(MoeContractTest, ExpertPagerIgnoresUnknownResidencyQueries) {
  us4::ExpertPager pager(1);
  pager.Touch("expert-a");

  EXPECT_FALSE(pager.IsResident("expert-missing"));
}
