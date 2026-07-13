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
  // Issue #81.7c: this fixture's expert0 shard has no gate/up/down_proj
  // tensors, so routing through a real FFN must be explicitly, visibly
  // false here -- not silently true just because the lm_head swap worked.
  EXPECT_FALSE(result.usedRealExpertFfn);

  // The base model's own lm_head.weight is a decoy tuned to argmax to
  // "alpha" (see generate_toy_moe_real.py); expert 0's real lm_head.weight
  // argmaxes to "beta" instead. Observing "beta" -- not "alpha" -- proves
  // the router's selected expert weight, not the base tensor, drove the
  // output.
  ASSERT_EQ(result.generatedTokens.size(), 1U);
  EXPECT_EQ(result.generatedTokens.front(), "beta");
}

// Issue #81.7c: beyond swapping the shared lm_head.weight (#81.7/#81.7b),
// the routed expert's real FFN layer (gate/up/down_proj SwiGLU) must
// transform the attention context before the output projection.
TEST(MoeRealExpertWeightsContractTest,
     RoutedExpertRealFfnTransformsContextBeforeProjection) {
  const us4::IUS4V6Adapter *adapter =
      us4::FindAdapterByModel("deepseek-v2-lite");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path tensorPath = RepoRoot() / "tests" / "fixtures" /
                                           "models" / "toy-moe-real-ffn" /
                                           "toy-moe-real-ffn.safetensors";
  ASSERT_TRUE(us4::LoadModelAsset(tensorPath, asset, &error)) << error;
  ASSERT_EQ(asset.expertShardPaths.size(), 2U);

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result = adapter->Generate(
      {.prompt = "", .maxTokens = 1, .asset = &asset}, context);

  ASSERT_TRUE(result.usedRealExpertWeights);
  ASSERT_TRUE(result.usedRealExpertFfn)
      << "expert 0's real gate/up/down_proj weights should have been "
         "loaded and applied to the attention context";

  // External oracle (see generate_toy_moe_real_ffn.py): the real,
  // single-token attention context is a one-hot vector on hidden dim 0;
  // expert 0's gate/up_proj zero out every FFN intermediate dim except
  // index 0, and down_proj routes that one surviving value to hidden dim
  // 3 -- so the FFN output is a positive scalar times one-hot(dim 3).
  // lm_head's column 3 argmaxes to "gamma", and a positive scale can't
  // flip that argmax.
  ASSERT_EQ(result.generatedTokens.size(), 1U);
  EXPECT_EQ(result.generatedTokens.front(), "gamma");
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
