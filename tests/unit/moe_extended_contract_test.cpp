#include <gtest/gtest.h>

#include <cmath>

#include "moe/moe_telemetry.h"
#include "moe/router_metrics.h"
#include "moe/shard_loader.h"

TEST(MoeExtendedContractTest, EntropyZeroForPeakedDistribution) {
  std::vector<float> logits = {100.0F, 0.0F, 0.0F};
  EXPECT_NEAR(us4::RouterEntropy(logits), 0.0F, 1e-3F);
}

TEST(MoeExtendedContractTest, EntropyMatchesUniformReference) {
  std::vector<float> logits = {0.0F, 0.0F, 0.0F, 0.0F};
  const float expected = std::log(4.0F);
  EXPECT_NEAR(us4::RouterEntropy(logits), expected, 1e-3F);
}

TEST(MoeExtendedContractTest, LoadBalanceLossIsZeroWhenUniformAndKMatchesExperts) {
  std::vector<float> logits = {0.0F, 0.0F, 0.0F, 0.0F};
  EXPECT_NEAR(us4::RouterLoadBalanceLoss(logits, 4U), 0.0F, 1e-3F);
}

TEST(MoeExtendedContractTest, RoutingTelemetryAttachesTopKDecision) {
  const auto telemetry = us4::ComputeRoutingTelemetry({0.1F, 1.5F, 0.2F, 0.4F}, 2);
  ASSERT_EQ(telemetry.topK.size(), 2U);
  EXPECT_EQ(telemetry.topK[0].expert, 1U);
  EXPECT_GE(telemetry.entropy, 0.0F);
}

TEST(MoeExtendedContractTest, ShardManifestParserAcceptsValidBody) {
  const std::string manifest =
      "family=deepseek\n"
      "model_id=deepseek-v3-moe\n"
      "[shard]\n"
      "expert_index=0\n"
      "shard_index=0\n"
      "shard_count=2\n"
      "file=experts/expert-0.safetensors\n"
      "weight_format=safetensors\n"
      "routed_only=false\n"
      "[shard]\n"
      "expert_index=1\n"
      "shard_index=0\n"
      "shard_count=2\n"
      "file=experts/expert-1.safetensors\n"
      "weight_format=safetensors\n"
      "routed_only=true\n";

  const auto parsed = us4::ParseExpertShardManifestBody(manifest);
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 2U);
  EXPECT_EQ((*parsed)[0].family, "deepseek");
  EXPECT_EQ((*parsed)[0].expertIndex, 0U);
  EXPECT_FALSE((*parsed)[0].routedOnly);
  EXPECT_TRUE((*parsed)[1].routedOnly);
  EXPECT_EQ((*parsed)[1].weightFormat, "safetensors");
}

TEST(MoeExtendedContractTest, ShardManifestParserRejectsMissingFields) {
  const std::string manifest =
      "family=deepseek\n"
      "[shard]\n"
      "expert_index=0\n";
  EXPECT_FALSE(us4::ParseExpertShardManifestBody(manifest).has_value());
}

TEST(MoeExtendedContractTest, TelemetrySnapshotPreservesEvents) {
  std::vector<us4::ExpertResidencyEvent> events = {
      {"expert-0", true, false, 12U},
      {"expert-3", false, true, 0U},
  };
  const auto snapshot = us4::BuildMoeTelemetrySnapshot(
      2U, 3U, 1U, 0.5F, 0.1F, 0.8F, std::move(events));

  ASSERT_EQ(snapshot.events.size(), 2U);
  EXPECT_EQ(snapshot.events[0].expertId, "expert-0");
  EXPECT_EQ(snapshot.events[1].expertId, "expert-3");
  EXPECT_FLOAT_EQ(snapshot.routerEntropy, 0.5F);
  EXPECT_FLOAT_EQ(snapshot.prefetchHitRatio, 0.8F);
}
