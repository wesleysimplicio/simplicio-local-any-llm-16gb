#include <gtest/gtest.h>

#include "cache/multimodal_cache.h"
#include "cache/sparsity_aware_cache.h"
#include "moe/speculative_prefetch.h"

TEST(MoeAdvancedContractTest, SpeculativePrefetchPredictsMostFrequent) {
  us4::SpeculativePrefetch predictor;
  predictor.Observe("expert-0", "expert-1");
  predictor.Observe("expert-0", "expert-1");
  predictor.Observe("expert-0", "expert-2");

  const auto predicted = predictor.Predict("expert-0", 1U);
  ASSERT_EQ(predicted.size(), 1U);
  EXPECT_EQ(predicted[0], "expert-1");
}

TEST(MoeAdvancedContractTest, SpeculativePrefetchExposesHitRatio) {
  us4::SpeculativePrefetch predictor;
  predictor.Observe("expert-0", "expert-1");
  predictor.Observe("expert-0", "expert-1");

  auto predicted = predictor.Predict("expert-0", 2U);
  predictor.RecordOutcome(predicted, "expert-1");
  EXPECT_EQ(predictor.LastPrefetchAttempts(), predicted.size());
  EXPECT_GE(predictor.LastPrefetchHits(), 1U);
  EXPECT_GT(predictor.HitRatio(), 0.0F);
}

TEST(MoeAdvancedContractTest, SparsityCacheHitMissAndEviction) {
  us4::SparsityAwareCache cache(2);
  cache.Store("a", {1.0F, 2.0F});
  cache.Store("b", {3.0F});
  EXPECT_EQ(cache.EntryCount(), 2U);
  EXPECT_TRUE(cache.Lookup("a").has_value());
  EXPECT_FALSE(cache.Lookup("missing").has_value());

  cache.Store("c", {4.0F, 5.0F});
  EXPECT_EQ(cache.EntryCount(), 2U);
}

TEST(MoeAdvancedContractTest, MultimodalCacheIsolatesByModality) {
  us4::MultimodalCache cache;
  us4::MultimodalCacheKey imageKey{"asset-1", us4::MultimodalModality::kImage, 0};
  us4::MultimodalCacheKey audioKey{"asset-1", us4::MultimodalModality::kAudio, 0};

  cache.Store(imageKey, {0.1F, 0.2F});
  cache.Store(audioKey, {0.9F});

  ASSERT_TRUE(cache.Lookup(imageKey).has_value());
  EXPECT_FLOAT_EQ(cache.Lookup(imageKey)->at(0), 0.1F);
  ASSERT_TRUE(cache.Lookup(audioKey).has_value());
  EXPECT_FLOAT_EQ(cache.Lookup(audioKey)->at(0), 0.9F);
}

TEST(MoeAdvancedContractTest, MultimodalCacheReportsMissForUnknownKey) {
  us4::MultimodalCache cache;
  us4::MultimodalCacheKey key{"asset-x", us4::MultimodalModality::kImage, 0};
  EXPECT_FALSE(cache.Lookup(key).has_value());
  EXPECT_EQ(cache.MissCount(), 1U);
}
