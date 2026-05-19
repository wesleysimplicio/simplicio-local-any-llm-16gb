#include <gtest/gtest.h>

#include "core/hardware_probe.h"
#include "tuning/auto_tuner.h"
#include "tuning/profile_cache.h"

TEST(TuningContractTest, AutoTunerPicksLowestLatencyCandidate) {
  us4::HardwareProbeResult hardware;
  hardware.chip = "M3";
  const std::vector<us4::AutoTunerCandidate> candidates = {
      {4U, 16U, 1U, 9.5F},
      {8U, 8U, 2U, 4.2F},
      {16U, 16U, 4U, 6.1F},
  };
  const auto profile = us4::SelectAutoTunerProfile(hardware, candidates);
  EXPECT_EQ(profile.tileRows, 8U);
  EXPECT_EQ(profile.tileCols, 8U);
  EXPECT_EQ(profile.batchSize, 2U);
  EXPECT_FLOAT_EQ(profile.estimatedLatencyMs, 4.2F);
  EXPECT_EQ(profile.chip, "M3");
}

TEST(TuningContractTest, AutoTunerHandlesEmptyCandidates) {
  us4::HardwareProbeResult hardware;
  hardware.chip = "M5";
  const auto profile = us4::SelectAutoTunerProfile(hardware, {});
  EXPECT_EQ(profile.tileRows, 0U);
  EXPECT_EQ(profile.chip, "M5");
}

TEST(TuningContractTest, ProfileCacheStoresAndLooksUp) {
  us4::ProfileCache cache;
  us4::ProfileCacheKey key{"M3", "qwen-0.5b"};
  us4::AutoTunerProfile profile;
  profile.chip = "M3";
  profile.tileRows = 8U;
  profile.tileCols = 16U;
  profile.batchSize = 4U;
  profile.estimatedLatencyMs = 3.1F;
  cache.Store(key, profile);

  EXPECT_EQ(cache.Size(), 1U);
  const auto retrieved = cache.Lookup(key);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved->batchSize, 4U);
}

TEST(TuningContractTest, ProfileCacheSerializeAndLoadRoundTrip) {
  us4::ProfileCache cache;
  us4::AutoTunerProfile profileA{"M3", 8U, 16U, 4U, 3.1F};
  us4::AutoTunerProfile profileB{"M5", 16U, 16U, 8U, 1.4F};
  cache.Store({"M3", "qwen-0.5b"}, profileA);
  cache.Store({"M5", "llama-3.1-8b"}, profileB);
  const std::string serialised = cache.Serialize();

  us4::ProfileCache restored;
  ASSERT_TRUE(restored.Load(serialised));
  EXPECT_EQ(restored.Size(), 2U);
  const auto roundTrip = restored.Lookup({"M5", "llama-3.1-8b"});
  ASSERT_TRUE(roundTrip.has_value());
  EXPECT_EQ(roundTrip->tileRows, 16U);
  EXPECT_EQ(roundTrip->batchSize, 8U);
}

TEST(TuningContractTest, ProfileCacheRejectsMalformedBody) {
  us4::ProfileCache cache;
  EXPECT_FALSE(cache.Load("not-a-real-entry\n"));
}
