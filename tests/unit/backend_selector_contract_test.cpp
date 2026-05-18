#include <gtest/gtest.h>

#include "adapters/llama/llama_adapter.h"
#include "adapters/qwen/qwen_adapter.h"
#include "core/backend_selector.h"

namespace {

us4::HardwareProbeResult MakeProbe() {
  us4::HardwareProbeResult probe;
  probe.platform = "macos";
  probe.architecture = "arm64";
  probe.chip = "apple-m";
  probe.unifiedMemoryGiB = 32;
  probe.recommendedMode = us4::RuntimeMode::kBalancedPlus;
  return probe;
}

} // namespace

TEST(BackendSelectorContractTest, ParsesRequestedBackendAliases) {
  EXPECT_EQ(us4::ParseBackendType("CPU"), us4::BackendType::kScalarCpu);
  EXPECT_EQ(us4::ParseBackendType("metal"), us4::BackendType::kMetal);
  EXPECT_EQ(us4::ParseBackendType("MLX"), us4::BackendType::kMlx);
  EXPECT_EQ(us4::ParseBackendType("neon"), us4::BackendType::kNeon);
  EXPECT_EQ(us4::ParseBackendType("ane"), us4::BackendType::kAne);
  EXPECT_FALSE(us4::ParseBackendType("cuda").has_value());
}

TEST(BackendSelectorContractTest, AutoSelectionPrefersAneForFullModeOnM5Hosts) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.chip = "Apple M5";
  probe.hasAne = true;
  probe.supportsCoreMl = true;
  probe.hasMetal = true;
  probe.hasMlx = true;
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasPerformanceCores = true;
  probe.hasEfficiencyCores = true;

  const us4::LlamaAdapter adapter;
  const us4::BackendSelection selection =
      us4::SelectBackend(probe, us4::RuntimeMode::kFull, adapter);

  EXPECT_EQ(selection.selected, us4::BackendType::kAne);
  EXPECT_FALSE(selection.fellBack);
  EXPECT_EQ(selection.reason, "auto-ane");
}

TEST(BackendSelectorContractTest,
     AutoSelectionPrefersMetalBeforeOtherBackends) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.hasMetal = true;
  probe.hasMlx = true;
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasPerformanceCores = true;
  probe.hasEfficiencyCores = true;

  const us4::LlamaAdapter adapter;
  const us4::BackendSelection selection =
      us4::SelectBackend(probe, us4::RuntimeMode::kBalancedPlus, adapter);

  EXPECT_EQ(selection.selected, us4::BackendType::kMetal);
  EXPECT_FALSE(selection.fellBack);
  EXPECT_EQ(selection.reason, "auto-metal");
}

TEST(BackendSelectorContractTest,
     AutoSelectionPrefersMlxWhenMetalModeIsDisallowed) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.hasMetal = true;
  probe.hasMlx = true;
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasPerformanceCores = true;
  probe.hasEfficiencyCores = true;

  const us4::LlamaAdapter adapter;
  const us4::BackendSelection selection =
      us4::SelectBackend(probe, us4::RuntimeMode::kDegraded, adapter);

  EXPECT_EQ(selection.selected, us4::BackendType::kMlx);
  EXPECT_FALSE(selection.fellBack);
  EXPECT_EQ(selection.reason, "auto-mlx");
}

TEST(BackendSelectorContractTest,
     AutoSelectionFallsToNeonForLowModesWithoutMlx) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.hasMetal = true;
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasEfficiencyCores = true;

  const us4::QwenAdapter adapter;
  const us4::BackendSelection selection =
      us4::SelectBackend(probe, us4::RuntimeMode::kMicroPlus, adapter);

  EXPECT_EQ(selection.selected, us4::BackendType::kNeon);
  EXPECT_FALSE(selection.fellBack);
  EXPECT_EQ(selection.reason, "auto-neon");
}

TEST(BackendSelectorContractTest,
     RequestedUnavailableBackendFallsBackToPreferredBackend) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasPerformanceCores = true;

  const us4::QwenAdapter adapter;
  const us4::BackendSelection selection = us4::SelectBackend(
      probe, us4::RuntimeMode::kDegraded, adapter, us4::BackendType::kMetal);

  EXPECT_EQ(selection.selected, us4::BackendType::kNeon);
  EXPECT_TRUE(selection.fellBack);
  EXPECT_EQ(selection.reason, "requested-backend-unavailable");
}

TEST(BackendSelectorContractTest, RequestedMlxFallsBackWhenModeDisallowsIt) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.hasMlx = true;
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasEfficiencyCores = true;

  const us4::LlamaAdapter adapter;
  const us4::BackendSelection selection = us4::SelectBackend(
      probe, us4::RuntimeMode::kMicro, adapter, us4::BackendType::kMlx);

  EXPECT_EQ(selection.selected, us4::BackendType::kNeon);
  EXPECT_TRUE(selection.fellBack);
  EXPECT_EQ(selection.reason, "requested-backend-unavailable");
}

TEST(BackendSelectorContractTest, RequestedAneFallsBackWhenModeDisallowsIt) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.chip = "Apple M5";
  probe.hasAne = true;
  probe.supportsCoreMl = true;
  probe.hasMetal = true;
  probe.hasMlx = true;
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasPerformanceCores = true;
  probe.hasEfficiencyCores = true;

  const us4::LlamaAdapter adapter;
  const us4::BackendSelection selection = us4::SelectBackend(
      probe, us4::RuntimeMode::kBalancedPlus, adapter, us4::BackendType::kAne);

  EXPECT_EQ(selection.selected, us4::BackendType::kMetal);
  EXPECT_TRUE(selection.fellBack);
  EXPECT_EQ(selection.reason, "requested-backend-unavailable");
}

TEST(BackendSelectorContractTest,
     NeonRequiresVectorWidthAndEligibleCpuCluster) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.hasNeon = true;
  probe.neonVectorBits = 64;
  probe.hasPerformanceCores = true;

  const us4::QwenAdapter adapter;
  const us4::BackendSelection narrowSelection =
      us4::SelectBackend(probe, us4::RuntimeMode::kMicroPlus, adapter);
  EXPECT_EQ(narrowSelection.selected, us4::BackendType::kScalarCpu);
  EXPECT_EQ(narrowSelection.reason, "auto-scalar");

  probe.neonVectorBits = 128;
  probe.hasPerformanceCores = false;
  probe.hasEfficiencyCores = true;
  const us4::BackendSelection degradedSelection =
      us4::SelectBackend(probe, us4::RuntimeMode::kDegraded, adapter);
  EXPECT_EQ(degradedSelection.selected, us4::BackendType::kScalarCpu);
  EXPECT_EQ(degradedSelection.reason, "auto-scalar");
}
