#include <gtest/gtest.h>

#include "kv/kv_pager.h"
#include "kv/prefix_cache.h"
#include "kv/summarizer.h"

TEST(KvContractTest, PrefixCacheTracksReferenceCountsUntilLastRelease) {
  us4::PrefixCache cache;

  cache.Retain("hello");
  cache.Retain("hello");

  const auto retained = cache.Lookup("hello");
  ASSERT_TRUE(retained.has_value());
  EXPECT_EQ(retained->refCount, 2U);
  EXPECT_EQ(cache.EntryCount(), 1U);

  cache.Release("hello");
  const auto afterFirstRelease = cache.Lookup("hello");
  ASSERT_TRUE(afterFirstRelease.has_value());
  EXPECT_EQ(afterFirstRelease->refCount, 1U);

  cache.Release("hello");
  EXPECT_FALSE(cache.Lookup("hello").has_value());
  EXPECT_EQ(cache.EntryCount(), 0U);
}

TEST(KvContractTest, SummarizerProducesMeanAndHandlesEmptyInput) {
  us4::Summarizer summarizer;

  EXPECT_TRUE(summarizer.Summarize({}).empty());

  const auto summary = summarizer.Summarize({2.0F, 4.0F, 6.0F, 8.0F});
  ASSERT_EQ(summary.size(), 1U);
  EXPECT_FLOAT_EQ(summary[0], 5.0F);
}

TEST(KvContractTest, PagerLookupTouchesPagesAndPreservesStoredValues) {
  us4::KvPager pager(1);
  pager.Append("prompt-a", {1.0F, 2.0F});
  pager.Append("prompt-b", {3.0F, 4.0F});

  EXPECT_EQ(pager.PageCount(), 2U);
  EXPECT_EQ(pager.HotPageCount(), 1U);

  const auto lookedUp = pager.Lookup("prompt-a");
  ASSERT_TRUE(lookedUp.has_value());
  EXPECT_EQ(lookedUp->key, "prompt-a");
  EXPECT_EQ(lookedUp->values.size(), 2U);
  EXPECT_FLOAT_EQ(lookedUp->values[0], 1.0F);
  EXPECT_FLOAT_EQ(lookedUp->values[1], 2.0F);
  EXPECT_EQ(lookedUp->tier, us4::KvTier::kWarm);
  EXPECT_EQ(lookedUp->hitCount, 1U);
}

TEST(KvContractTest, PagerReturnsMissingForUnknownKeys) {
  us4::KvPager pager;
  pager.Append("known", {1.0F});

  EXPECT_FALSE(pager.Lookup("missing").has_value());
}
