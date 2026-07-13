#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "adapters/adapter_registry.h"
#include "core/hardware_probe.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"

namespace {

us4::HardwareProbeResult MakeProbe() {
  us4::HardwareProbeResult probe;
  probe.platform = "macos";
  probe.architecture = "arm64";
  probe.chip = "apple-m";
  probe.unifiedMemoryGiB = 24;
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasPerformanceCores = true;
  probe.hasEfficiencyCores = true;
  probe.recommendedMode = us4::RuntimeMode::kDegraded;
  return probe;
}

std::filesystem::path RepoRoot() {
#ifdef US4_SOURCE_DIR
  return std::filesystem::path(US4_SOURCE_DIR);
#else
  return std::filesystem::path(__FILE__)
      .parent_path()
      .parent_path()
      .parent_path();
#endif
}

} // namespace

// Issue #81.7: the router's selected expert must change what the forward
// actually computes, not just what the expert pager logs as "touched".
TEST(MoeRealExpertWeightsContractTest,
     RoutedExpertRealWeightOverridesBaseDecoyOutput) {
  const us4::IUS4V6Adapter *adapter =
      us4::FindAdapterByModel("deepseek-v2-lite");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path tensorPath = RepoRoot() / "tests" / "fixtures" /
                                           "models" / "toy-moe-real" /
                                           "toy-moe-real.safetensors";
  ASSERT_TRUE(us4::LoadModelAsset(tensorPath, asset, &error)) << error;
  ASSERT_EQ(asset.expertShardPaths.size(), 2U);

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result = adapter->Generate(
      {.prompt = "", .maxTokens = 1, .asset = &asset}, context);

  ASSERT_TRUE(result.usedRealExpertWeights)
      << "expert 0's real lm_head.weight should have been loaded and "
         "applied to the forward";

  // The base model's own lm_head.weight is a decoy tuned to argmax to
  // "alpha" (see generate_toy_moe_real.py); expert 0's real lm_head.weight
  // argmaxes to "beta" instead. Observing "beta" -- not "alpha" -- proves
  // the router's selected expert weight, not the base tensor, drove the
  // output.
  ASSERT_EQ(result.generatedTokens.size(), 1U);
  EXPECT_EQ(result.generatedTokens.front(), "beta");
}

// Issue #81.7b: the deepseek-only wiring from #81.7/#88 must also cover the
// other MoE adapters that share DenseAdapterBase::Generate -- kimi, minimax,
// and glm each get their own fixture set so a regression in one family's
// wiring cannot hide behind another family's fixture.
TEST(MoeRealExpertWeightsContractTest,
     KimiRoutedExpertRealWeightOverridesBaseDecoyOutput) {
  const us4::IUS4V6Adapter *adapter =
      us4::FindAdapterByModel("kimi-k2-instruct");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path tensorPath = RepoRoot() / "tests" / "fixtures" /
                                           "models" / "toy-moe-real-kimi" /
                                           "toy-moe-real-kimi.safetensors";
  ASSERT_TRUE(us4::LoadModelAsset(tensorPath, asset, &error)) << error;
  ASSERT_EQ(asset.expertShardPaths.size(), 2U);

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result = adapter->Generate(
      {.prompt = "", .maxTokens = 1, .asset = &asset}, context);

  ASSERT_TRUE(result.usedRealExpertWeights)
      << "expert 0's real lm_head.weight should have been loaded and "
         "applied to the forward";
  ASSERT_EQ(result.generatedTokens.size(), 1U);
  EXPECT_EQ(result.generatedTokens.front(), "gamma");
}

TEST(MoeRealExpertWeightsContractTest,
     MiniMaxRoutedExpertRealWeightOverridesBaseDecoyOutput) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("minimax-m2");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path tensorPath = RepoRoot() / "tests" / "fixtures" /
                                           "models" / "toy-moe-real-minimax" /
                                           "toy-moe-real-minimax.safetensors";
  ASSERT_TRUE(us4::LoadModelAsset(tensorPath, asset, &error)) << error;
  ASSERT_EQ(asset.expertShardPaths.size(), 2U);

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result = adapter->Generate(
      {.prompt = "", .maxTokens = 1, .asset = &asset}, context);

  ASSERT_TRUE(result.usedRealExpertWeights)
      << "expert 0's real lm_head.weight should have been loaded and "
         "applied to the forward";
  ASSERT_EQ(result.generatedTokens.size(), 1U);
  EXPECT_EQ(result.generatedTokens.front(), "delta");
}

TEST(MoeRealExpertWeightsContractTest,
     GlmRoutedExpertRealWeightOverridesBaseDecoyOutput) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("glm-5.1");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path tensorPath = RepoRoot() / "tests" / "fixtures" /
                                           "models" / "toy-moe-real-glm" /
                                           "toy-moe-real-glm.safetensors";
  ASSERT_TRUE(us4::LoadModelAsset(tensorPath, asset, &error)) << error;
  ASSERT_EQ(asset.expertShardPaths.size(), 3U);

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result = adapter->Generate(
      {.prompt = "", .maxTokens = 1, .asset = &asset}, context);

  ASSERT_TRUE(result.usedRealExpertWeights)
      << "expert 0's real lm_head.weight should have been loaded and "
         "applied to the forward";
  ASSERT_EQ(result.generatedTokens.size(), 1U);
  EXPECT_EQ(result.generatedTokens.front(), "beta");
}
