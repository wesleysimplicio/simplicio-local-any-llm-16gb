#include <gtest/gtest.h>

#include "adapters/gemma/gemma_adapter.h"
#include "adapters/llama/llama_adapter.h"
#include "adapters/qwen/qwen_adapter.h"
#include "ane/ane_backend.h"
#include "core/runtime_context.h"
#include "memory/unified_allocator.h"
#include "metal/autorelease_scope.h"
#include "metal/command_queue.h"
#include "metal/dense_dispatch.h"
#include "metal/device_info.h"
#include "metal/kernel_library.h"
#include "mlx/dense_plan.h"
#include "mlx/mlx_bridge.h"

namespace {

us4::HardwareProbeResult MakeAppleProbe() {
  us4::HardwareProbeResult probe;
  probe.platform = "macos";
  probe.architecture = "arm64";
  probe.chip = "apple-m";
  probe.unifiedMemoryGiB = 36;
  probe.hasMetal = true;
  probe.hasMlx = true;
  probe.hasNeon = true;
  probe.recommendedMode = us4::RuntimeMode::kBalancedPlus;
  return probe;
}

} // namespace

TEST(RuntimeAccelerationContractTest, UnifiedAllocatorTracksSharedAllocations) {
  us4::UnifiedAllocator allocator;
  allocator.Allocate(64, false);
  allocator.Allocate(256, true);

  EXPECT_EQ(allocator.AllocationCount(), 2U);
  EXPECT_EQ(allocator.SharedAllocationCount(), 1U);
  EXPECT_EQ(allocator.ResidentBytes(), 320U);
}

TEST(RuntimeAccelerationContractTest,
     RuntimeContextExposesMetalQueueAndMlxBridge) {
  us4::RuntimeContext context(MakeAppleProbe());

  EXPECT_TRUE(context.metalQueue().Available());
  EXPECT_EQ(context.metalQueue().Reason(), "metal-queue-ready");
  EXPECT_EQ(context.metalQueue().Profile().stage,
            us4::MetalInitializationStage::kQueueReady);
  EXPECT_TRUE(context.metalQueue().Profile().queueCreated);
  EXPECT_TRUE(context.metalQueue().Profile().requiresAutoreleaseBoundary);
  EXPECT_EQ(context.metalQueue().Device().queueLabel, "us4.metal.default");
  EXPECT_TRUE(context.metalQueue().Device().supportsUnifiedMemory);
  EXPECT_TRUE(context.mlxBridge().Available());
  EXPECT_EQ(context.mlxBridge().Reason(), "mlx-bridge-ready");
  EXPECT_FALSE(context.aneBackend().Available());
  EXPECT_EQ(context.aneBackend().Reason(), "ane-unavailable");
}

TEST(RuntimeAccelerationContractTest, AneBackendCompilesAndPredictsOnM5Probe) {
  us4::HardwareProbeResult probe = MakeAppleProbe();
  probe.chip = "Apple M5";
  probe.hasAne = true;
  probe.supportsCoreMl = true;
  probe.recommendedMode = us4::RuntimeMode::kFull;

  us4::RuntimeContext context(probe);
  EXPECT_TRUE(context.aneBackend().Available());
  EXPECT_EQ(context.aneBackend().Reason(), "ane-backend-ready");

  const us4::AneCompilePlan plan{
      .kind = us4::AneModelKind::kAttentionMlp,
      .family = "llama",
      .layerName = "decoder.block.0.mlp",
      .tokenCount = 8U,
      .usesSharedTokenizer = true,
      .staticShapePreferred = true,
  };
  EXPECT_TRUE(context.aneBackend().Compile(plan));
  ASSERT_TRUE(context.aneBackend().LastCompiledModel().has_value());
  EXPECT_EQ(context.aneBackend().LastCompiledModel()->family, "llama");
  EXPECT_EQ(context.aneBackend().LastCompiledModel()->computeUnits, "ane-only");
  EXPECT_TRUE(
      context.aneBackend().LastCompiledModel()->usedCoreMlCompileIntent);
  EXPECT_TRUE(context.aneBackend().Predict(3U, 1U));
  EXPECT_TRUE(context.aneBackend().LastPredictionSucceeded());
  EXPECT_EQ(context.aneBackend().PredictionCount(), 1U);
  EXPECT_EQ(context.aneBackend().Reason(), "ane-predict-recorded");
}

TEST(RuntimeAccelerationContractTest,
     MetalDeviceProbeAndAutoreleaseScopeStayCallable) {
  const us4::MetalDeviceInfo device = us4::ProbeMetalDevice(MakeAppleProbe());
  const us4::ScopedAutoreleasePool pool(true);

  EXPECT_TRUE(device.available);
  EXPECT_EQ(device.maxThreadsPerThreadgroup, 1024U);
  EXPECT_TRUE(pool.Requested());
  if (pool.Kind() == us4::AutoreleaseBoundaryKind::kObjectiveC) {
    EXPECT_TRUE(pool.Active());
  } else {
    EXPECT_FALSE(pool.Active());
  }
}

TEST(RuntimeAccelerationContractTest, MetalQueueRecordsSharedDispatches) {
  us4::RuntimeContext context(MakeAppleProbe());
  const auto shared = context.allocator().Allocate(512, true);

  EXPECT_TRUE(context.metalQueue().Dispatch(us4::MetalKernelKind::kMatmul, 4,
                                            32, shared));
  ASSERT_EQ(context.metalQueue().DispatchCount(), 1U);
  EXPECT_EQ(context.metalQueue().Dispatches().front().entryPoint,
            "us4_matmul_fp16");
  EXPECT_EQ(context.metalQueue().Dispatches().front().relativePath,
            "runtime/metal/kernels/matmul.metal");
  EXPECT_TRUE(context.metalQueue().Dispatches().front().usesSharedAllocation);
  EXPECT_TRUE(
      context.metalQueue().Dispatches().front().autoreleaseBoundaryRequested);
  EXPECT_EQ(context.metalQueue().Reason(), "metal-dispatch-recorded");
}

TEST(RuntimeAccelerationContractTest, MetalKernelCatalogExposesKernelMetadata) {
  const auto &catalog = us4::GetMetalKernelCatalog();

  ASSERT_EQ(catalog.size(), 3U);
  EXPECT_NE(us4::FindMetalKernel(us4::MetalKernelKind::kMatmul), nullptr);
  EXPECT_NE(us4::FindMetalKernel(us4::MetalKernelKind::kSoftmax), nullptr);
  EXPECT_NE(us4::FindMetalKernel(us4::MetalKernelKind::kRmsNorm), nullptr);
  EXPECT_EQ(catalog[1].entryPoint, "us4_softmax_rows");
  EXPECT_EQ(catalog[2].relativePath, "runtime/metal/kernels/rmsnorm.metal");
  EXPECT_FALSE(catalog[0].source.empty());
}

TEST(RuntimeAccelerationContractTest,
     DenseMetalDispatchPlanBuildsThreeStageSequence) {
  const us4::DenseMetalDispatchPlan plan =
      us4::BuildDenseMetalDispatchPlan(8, 8, 16);

  ASSERT_EQ(plan.steps.size(), 3U);
  EXPECT_EQ(plan.steps[0].kernel, us4::MetalKernelKind::kMatmul);
  EXPECT_EQ(plan.steps[1].kernel, us4::MetalKernelKind::kSoftmax);
  EXPECT_EQ(plan.steps[2].kernel, us4::MetalKernelKind::kRmsNorm);
}

TEST(RuntimeAccelerationContractTest, MlxBridgeBuildsAndEvaluatesDensePlan) {
  us4::RuntimeContext context(MakeAppleProbe());
  const auto shared = context.allocator().Allocate(1024, true);

  EXPECT_TRUE(context.mlxBridge().BuildDensePlan("llama", 64, shared));
  ASSERT_TRUE(context.mlxBridge().LastPlan().has_value());
  EXPECT_EQ(context.mlxBridge().LastPlan()->family, "llama");
  EXPECT_TRUE(context.mlxBridge().LastPlan()->usesUnifiedAllocation);
  ASSERT_EQ(context.mlxBridge().LastPlan()->operations.size(), 3U);
  EXPECT_EQ(context.mlxBridge().LastPlan()->operations[0].kind,
            us4::MlxOperationKind::kEmbeddingLookup);
  EXPECT_EQ(context.mlxBridge().LastPlan()->operations[1].kind,
            us4::MlxOperationKind::kAttention);
  EXPECT_EQ(context.mlxBridge().LastPlan()->operations[2].kind,
            us4::MlxOperationKind::kProjection);
  EXPECT_TRUE(context.mlxBridge().EvaluateLastPlan());
  EXPECT_TRUE(context.mlxBridge().LastEvaluationSucceeded());
  EXPECT_EQ(context.mlxBridge().Reason(), "mlx-plan-evaluated");
}

TEST(RuntimeAccelerationContractTest, MlxDensePlanBuildsThreeOperations) {
  const us4::MlxDensePlan plan = us4::BuildMlxDensePlan(8, 8, 16);

  ASSERT_EQ(plan.operations.size(), 3U);
  EXPECT_EQ(plan.operations[0].kind, us4::MlxOperationKind::kEmbeddingLookup);
  EXPECT_EQ(plan.operations[1].kind, us4::MlxOperationKind::kAttention);
  EXPECT_EQ(plan.operations[2].kind, us4::MlxOperationKind::kProjection);
}

TEST(RuntimeAccelerationContractTest,
     LlamaGenerationTouchesMetalScaffoldWhenSelected) {
  us4::RuntimeContext context(MakeAppleProbe());
  const us4::LlamaAdapter adapter;

  const us4::GenerationResult result =
      adapter.Generate({.prompt = "hello",
                        .maxTokens = 4,
                        .asset = nullptr,
                        .requestedBackend = us4::BackendType::kMetal},
                       context);

  EXPECT_EQ(result.backend, "metal");
  EXPECT_EQ(result.sharedAllocations, 1U);
  EXPECT_EQ(result.metalDispatches, 3U);
  EXPECT_FALSE(result.mlxPlanBuilt);
  EXPECT_EQ(context.metalQueue().DispatchCount(), 3U);
  EXPECT_EQ(context.allocator().SharedAllocationCount(), 1U);
}

TEST(RuntimeAccelerationContractTest, QwenAndGemmaDeclareMetalSupport) {
  const us4::QwenAdapter qwen;
  const us4::GemmaAdapter gemma;

  EXPECT_TRUE(qwen.SupportsMetalBackend());
  EXPECT_TRUE(gemma.SupportsMetalBackend());
}
