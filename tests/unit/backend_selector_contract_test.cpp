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

}  // namespace

TEST(BackendSelectorContractTest, ParsesRequestedBackendAliases) {
  EXPECT_EQ(us4::ParseBackendType("CPU"), us4::BackendType::kScalarCpu);
  EXPECT_EQ(us4::ParseBackendType("metal"), us4::BackendType::kMetal);
  EXPECT_EQ(us4::ParseBackendType("MLX"), us4::BackendType::kMlx);
  EXPECT_EQ(us4::ParseBackendType("neon"), us4::BackendType::kNeon);
  EXPECT_EQ(us4::ParseBackendType("ane"), us4::BackendType::kAne);
  EXPECT_FALSE(us4::ParseBackendType("cuda").has_value());
}

TEST(BackendSelectorContractTest, AutoSelectionPrefersMetalBeforeOtherBackends) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.hasMetal = true;
  probe.hasMlx = true;
  probe.hasNeon = true;

  const us4::LlamaAdapter adapter;
  const us4::BackendSelection selection =
      us4::SelectBackend(probe, us4::RuntimeMode::kBalancedPlus, adapter);

  EXPECT_EQ(selection.selected, us4::BackendType::kMetal);
  EXPECT_FALSE(selection.fellBack);
  EXPECT_EQ(selection.reason, "auto");
}

TEST(BackendSelectorContractTest, RequestedUnavailableBackendFallsBackToPreferredBackend) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.hasNeon = true;

  const us4::QwenAdapter adapter;
  const us4::BackendSelection selection =
      us4::SelectBackend(probe, us4::RuntimeMode::kDegraded, adapter, us4::BackendType::kMetal);

  EXPECT_EQ(selection.selected, us4::BackendType::kNeon);
  EXPECT_TRUE(selection.fellBack);
  EXPECT_EQ(selection.reason, "requested-backend-unavailable");
}

TEST(BackendSelectorContractTest, RequestedMlxFallsBackWhenModeDisallowsIt) {
  us4::HardwareProbeResult probe = MakeProbe();
  probe.hasMlx = true;
  probe.hasNeon = true;

  const us4::LlamaAdapter adapter;
  const us4::BackendSelection selection =
      us4::SelectBackend(probe, us4::RuntimeMode::kMicro, adapter, us4::BackendType::kMlx);

  EXPECT_EQ(selection.selected, us4::BackendType::kNeon);
  EXPECT_TRUE(selection.fellBack);
  EXPECT_EQ(selection.reason, "requested-backend-unavailable");
}
