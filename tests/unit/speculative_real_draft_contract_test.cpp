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

// Issue #81.9: the speculative draft proposal must come from a genuine
// forward over a real (smaller) draft model, not from copying the target's
// authoritative tokens and incrementing the last one.
TEST(SpeculativeRealDraftContractTest,
     DraftProposalComesFromRealDraftModelForwardNotMockedCopy) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("qwen-0.5b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path tensorPath = RepoRoot() / "tests" / "fixtures" /
                                           "models" / "toy-dense-real" /
                                           "toy-dense-real.safetensors";
  ASSERT_TRUE(us4::LoadModelAsset(tensorPath, asset, &error)) << error;
  ASSERT_FALSE(asset.draftModelPath.empty())
      << "fixture manifest must declare draft_model_path";

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result = adapter->Generate(
      {.prompt = "alpha", .maxTokens = 1, .asset = &asset}, context);

  ASSERT_TRUE(result.usedRealWeights);
  ASSERT_TRUE(result.usedRealDraftModel)
      << "draft model has real embedding/lm_head tensors shaped to match "
         "the shared vocabulary, so the real forward path must activate";

  // The target model's real forward for "alpha" argmaxes to "delta" (see
  // generate_toy_dense_real.py); the draft fixture's own real forward for
  // a previous token of "delta" ALSO argmaxes to "delta" (see
  // generate_toy_draft_model.py). Both facts were computed offline,
  // independently of this test, from the fixtures' actual tensor values --
  // so accepting all draft tokens here reflects a genuine computation
  // matching, not a scripted mock guaranteed to agree or disagree.
  EXPECT_EQ(result.text, "delta");
  EXPECT_EQ(result.speculativeAcceptedTokens, 1U);
  EXPECT_EQ(result.speculativeRejectedTokens, 0U);
  EXPECT_DOUBLE_EQ(result.speculativeAcceptanceRate, 1.0);
  EXPECT_EQ(result.speculativeLookaheadTokens, 1U);
  EXPECT_EQ(result.speculativeVerifyWindow, 1U);
  EXPECT_TRUE(result.speculativeWarmupActive);
  EXPECT_FALSE(result.speculativeMtpEnabled);
}
