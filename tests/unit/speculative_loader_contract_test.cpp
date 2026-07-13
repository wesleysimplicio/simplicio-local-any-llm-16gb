#include <gtest/gtest.h>

#include "speculative/draft_model_loader.h"
#include "speculative/speculative_telemetry.h"

TEST(DraftModelLoaderContractTest, ParsesCompleteManifest) {
  const std::string body = "# draft companion for llama\n"
                           "family=llama\n"
                           "model_id=llama-3.1-8b-draft\n"
                           "tokenizer_hash=abc123\n"
                           "file=draft-llama.gguf\n"
                           "weight_format=gguf\n";

  const auto descriptor = us4::ParseDraftModelManifestBody(body);

  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->family, "llama");
  EXPECT_EQ(descriptor->modelId, "llama-3.1-8b-draft");
  EXPECT_EQ(descriptor->tokenizerHash, "abc123");
  EXPECT_EQ(descriptor->filePath, "draft-llama.gguf");
  EXPECT_EQ(descriptor->weightFormat, "gguf");
}

TEST(DraftModelLoaderContractTest, RejectsManifestMissingRequiredField) {
  // Missing tokenizer_hash -- the verifier cannot trust draft tokens without
  // it.
  const std::string body = "family=llama\n"
                           "model_id=llama-3.1-8b-draft\n"
                           "file=draft-llama.gguf\n"
                           "weight_format=gguf\n";

  EXPECT_FALSE(us4::ParseDraftModelManifestBody(body).has_value());
}

TEST(DraftModelLoaderContractTest, IgnoresCommentsAndBlankLinesAndUnknownKeys) {
  const std::string body = "\n"
                           "  # comment\n"
                           "family = llama \n"
                           "unknown_key=whatever\n"
                           "model_id=draft\n"
                           "tokenizer_hash=h\n"
                           "file=f.gguf\n"
                           "weight_format=gguf\n";

  const auto descriptor = us4::ParseDraftModelManifestBody(body);
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->family, "llama");
  EXPECT_EQ(descriptor->modelId, "draft");
}

TEST(SpeculativeTelemetryContractTest, ComputesAcceptanceAndRejections) {
  const us4::SpeculativeTelemetry telemetry =
      us4::ComputeSpeculativeTelemetry(8U, 6U, 4U);

  EXPECT_EQ(telemetry.draftAttempts, 8U);
  EXPECT_EQ(telemetry.acceptedTokens, 6U);
  EXPECT_EQ(telemetry.rejectedTokens, 2U);
  EXPECT_EQ(telemetry.verifyWindow, 4U);
  EXPECT_FLOAT_EQ(telemetry.acceptanceRate, 0.75F);
}

TEST(SpeculativeTelemetryContractTest, ZeroAttemptsHasZeroRateAndNoUnderflow) {
  const us4::SpeculativeTelemetry telemetry =
      us4::ComputeSpeculativeTelemetry(0U, 0U, 0U);

  EXPECT_EQ(telemetry.rejectedTokens, 0U);
  EXPECT_FLOAT_EQ(telemetry.acceptanceRate, 0.0F);
}

TEST(SpeculativeTelemetryContractTest, AcceptedAboveAttemptsClampsRejections) {
  // Defensive: accepted should never exceed attempts, but the math must not
  // underflow the unsigned rejected counter if it does.
  const us4::SpeculativeTelemetry telemetry =
      us4::ComputeSpeculativeTelemetry(2U, 5U, 3U);

  EXPECT_EQ(telemetry.rejectedTokens, 0U);
}

TEST(SpeculativeTelemetryContractTest,
     AdaptivePlanStartsInWarmupAndExpandsAfterStableAcceptance) {
  us4::AdaptiveSpeculativeConfig config;
  config.warmupDrafts = 2U;
  config.minLookaheadTokens = 1U;
  config.maxLookaheadTokens = 4U;

  us4::AdaptiveSpeculativeState state;
  const us4::AdaptiveSpeculativePlan first =
      us4::PlanAdaptiveSpeculation(state, config);
  EXPECT_TRUE(first.warmupActive);
  EXPECT_EQ(first.lookaheadTokens, 1U);
  EXPECT_FALSE(first.mtpEnabled);

  us4::UpdateAdaptiveSpeculativeState(
      state, us4::ComputeSpeculativeTelemetry(1U, 1U, 1U), config);
  us4::UpdateAdaptiveSpeculativeState(
      state, us4::ComputeSpeculativeTelemetry(1U, 1U, 1U), config);

  const us4::AdaptiveSpeculativePlan warmed =
      us4::PlanAdaptiveSpeculation(state, config);
  EXPECT_FALSE(warmed.warmupActive);
  EXPECT_TRUE(warmed.mtpEnabled);
  EXPECT_GE(warmed.lookaheadTokens, 2U);
}

TEST(SpeculativeTelemetryContractTest,
     AdaptivePlanStaysConservativeWhenAcceptanceIsLow) {
  us4::AdaptiveSpeculativeConfig config;
  config.warmupDrafts = 1U;
  config.minLookaheadTokens = 1U;
  config.maxLookaheadTokens = 4U;
  config.minAcceptanceRateForMtp = 0.6F;

  us4::AdaptiveSpeculativeState state;
  us4::UpdateAdaptiveSpeculativeState(
      state, us4::ComputeSpeculativeTelemetry(4U, 1U, 1U), config);

  const us4::AdaptiveSpeculativePlan plan =
      us4::PlanAdaptiveSpeculation(state, config);
  EXPECT_FALSE(plan.warmupActive);
  EXPECT_FALSE(plan.mtpEnabled);
  EXPECT_EQ(plan.lookaheadTokens, 1U);
}
