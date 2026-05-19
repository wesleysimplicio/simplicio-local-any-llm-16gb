#include <gtest/gtest.h>

#include "ane/ane_backend.h"
#include "ane/layer_offloader.h"
#include "ane/mixed_dispatch.h"
#include "tuning/thermal_monitor.h"

namespace {

us4::HardwareProbeResult MakeM5Hardware() {
  us4::HardwareProbeResult hardware;
  hardware.architecture = "arm64";
  hardware.isAppleSilicon = true;
  hardware.hasAne = true;
  hardware.chip = "M5";
  return hardware;
}

us4::HardwareProbeResult MakeM3Hardware() {
  us4::HardwareProbeResult hardware;
  hardware.architecture = "arm64";
  hardware.isAppleSilicon = true;
  hardware.hasAne = false;
  hardware.chip = "M3 Max";
  return hardware;
}

} // namespace

TEST(AneContractTest, BackendReportsReadyOnM5) {
  us4::AneBackend backend;
  const auto readiness = backend.Probe(MakeM5Hardware());
  EXPECT_TRUE(readiness.available);
  EXPECT_EQ(readiness.fallbackReason, "ready");
}

TEST(AneContractTest, BackendFallsBackOnM3) {
  us4::AneBackend backend;
  const auto readiness = backend.Probe(MakeM3Hardware());
  EXPECT_FALSE(readiness.available);
  EXPECT_EQ(readiness.fallbackReason, "chip-too-old");
}

TEST(AneContractTest, BackendReportsDeferredCompile) {
  us4::AneBackend backend;
  std::string reason;
  EXPECT_FALSE(backend.CompileForOffload("llama-3.1-8b", reason));
  EXPECT_EQ(reason, "compile-deferred");
}

TEST(AneContractTest, LayerOffloadPlanFallsBackToMetalWhenAneUnavailable) {
  const std::vector<us4::LayerDescriptor> layers = {
      {"attn-0", us4::LayerKind::kAttention, true},
      {"mlp-0", us4::LayerKind::kMlp, true},
  };
  const auto plan = us4::PlanLayerOffload(layers, /*aneAvailable=*/false);
  EXPECT_TRUE(plan.aneLayers.empty());
  EXPECT_EQ(plan.metalLayers.size(), 2U);
}

TEST(AneContractTest, LayerOffloadPlanPicksStaticLayersOnly) {
  const std::vector<us4::LayerDescriptor> layers = {
      {"attn-0", us4::LayerKind::kAttention, true},
      {"kv-0", us4::LayerKind::kStateful, true},
      {"mlp-0", us4::LayerKind::kMlp, false},
  };
  const auto plan = us4::PlanLayerOffload(layers, /*aneAvailable=*/true);
  ASSERT_EQ(plan.aneLayers.size(), 1U);
  EXPECT_EQ(plan.aneLayers[0], "attn-0");
  EXPECT_EQ(plan.rejectedLayers.size(), 1U);
}

TEST(AneContractTest, MixedDispatchAttributesBackendForEachStep) {
  us4::LayerOffloadPlan offload;
  offload.aneLayers = {"attn-0"};
  offload.metalLayers = {"mlp-0"};
  const auto plan = us4::BuildMixedDispatchPlan(offload);
  ASSERT_EQ(plan.steps.size(), 2U);
  EXPECT_EQ(plan.steps[0].backend, "ane");
  EXPECT_EQ(plan.steps[1].backend, "metal");
}

TEST(AneContractTest, ThermalMonitorTriggersDowngradeOnSerious) {
  us4::ThermalReading reading;
  reading.state = us4::ThermalState::kSerious;
  reading.celsius = 92.0F;
  const auto decision = us4::DecideThermalDowngrade(reading);
  EXPECT_TRUE(decision.requiresDowngrade);
  EXPECT_EQ(decision.reason, "serious-thermal");
}

TEST(AneContractTest, ThermalMonitorReportsNominal) {
  us4::ThermalReading reading;
  reading.state = us4::ThermalState::kNominal;
  const auto decision = us4::DecideThermalDowngrade(reading);
  EXPECT_FALSE(decision.requiresDowngrade);
}
