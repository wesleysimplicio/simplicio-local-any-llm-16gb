#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <string_view>
#include <vector>

#include "adapters/adapter_registry.h"
#include "adapters/llama/llama_adapter.h"
#include "adapters/llama/llama_config.h"
#include "adapters/qwen/qwen_adapter.h"
#include "ane/ane_backend.h"
#include "ane/layer_offloader.h"
#include "ane/mixed_dispatch.h"
#include "cache/multimodal_cache.h"
#include "cache/sparsity_aware_cache.h"
#include "core/backend_selector.h"
#include "core/gqa_attention.h"
#include "core/model_asset.h"
#include "core/rope.h"
#include "core/runtime_context.h"
#include "core/tensor.h"
#include "cpu/scalar_attention.h"
#include "cpu/scalar_matmul.h"
#include "kv/kv_pager.h"
#include "kv/prefix_cache.h"
#include "kv/summarizer.h"
#include "memory/unified_allocator.h"
#include "metal/autorelease_scope.h"
#include "metal/command_queue.h"
#include "metal/dense_dispatch.h"
#include "metal/device_info.h"
#include "metal/kernel_library.h"
#include "mlx/dense_plan.h"
#include "mlx/mlx_bridge.h"
#include "moe/expert_pager.h"
#include "moe/router.h"
#include "moe/speculative_prefetch.h"
#include "neon/dequant_int4.h"
#include "neon/dequant_int8.h"
#include "neon/kernel_profile.h"
#include "neon/neon_attention.h"
#include "neon/neon_matmul.h"
#include "scheduler/continuous_batcher.h"
#include "scheduler/session_pool.h"
#include "speculative/eagle3_decoder.h"
#include "speculative/peagle_decoder.h"
#include "sprint_01_contract_placeholders.h"
#include "tuning/thermal_monitor.h"

namespace {

bool Expect(const bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

us4::HardwareProbeResult MakeProbe() {
  us4::HardwareProbeResult probe;
  probe.platform = "macos";
  probe.architecture = "arm64";
  probe.chip = "apple-m";
  probe.unifiedMemoryGiB = 24;
  probe.isAppleSilicon = true;
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

void FillHalfTensor(us4::Tensor &tensor, const std::vector<float> &values,
                    const bool bfloat16) {
  std::uint16_t *data = tensor.MutableDataAsUInt16();
  if (data == nullptr || values.size() != tensor.ElementCount()) {
    return;
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = bfloat16 ? us4::EncodeBFloat16(values[index])
                           : us4::EncodeFloat16(values[index]);
  }
}

float QuantizeHalfValue(const float value, const bool bfloat16) {
  return bfloat16 ? us4::DecodeBFloat16(us4::EncodeBFloat16(value))
                  : us4::DecodeFloat16(us4::EncodeFloat16(value));
}

bool FillHalfReferenceTensor(us4::Tensor &tensor,
                             const std::vector<float> &values,
                             const bool bfloat16) {
  float *data = tensor.MutableDataAsFloat32();
  if (data == nullptr || values.size() != tensor.ElementCount()) {
    return false;
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = QuantizeHalfValue(values[index], bfloat16);
  }
  return true;
}

void FillFloatTensor(us4::Tensor &tensor, const std::vector<float> &values) {
  float *data = tensor.MutableDataAsFloat32();
  if (data == nullptr || values.size() != tensor.ElementCount()) {
    return;
  }

  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = values[index];
  }
}

} // namespace

int main() {
  bool ok = true;

  {
    constexpr auto &modes = sprint_01_contract::kCanonicalRuntimeModes;
    ok &= Expect(modes.size() == 7U,
                 "runtime mode contract should keep 7 canonical modes");
    std::set<int> seen;
    for (const auto mode : modes) {
      ok &= Expect(seen.insert(static_cast<int>(mode)).second,
                   "runtime mode values should stay unique");
    }
  }

  {
    constexpr auto &responsibilities =
        sprint_01_contract::kProbeResponsibilities;
    ok &= Expect(responsibilities.size() == 5U,
                 "hardware probe contract should keep 5 responsibilities");
    std::set<std::string_view> seen;
    for (const auto &responsibility : responsibilities) {
      ok &= Expect(seen.insert(responsibility.name).second,
                   "hardware responsibilities should stay unique");
      ok &= Expect(!responsibility.description.empty(),
                   "hardware responsibility description should not be empty");
    }
  }

  {
    constexpr auto &categories = sprint_01_contract::kTelemetryCategories;
    ok &= Expect(categories.size() == 6U,
                 "telemetry contract should keep 6 categories");
  }

  {
    const auto cpu = us4::ParseBackendType("CPU");
    const auto metal = us4::ParseBackendType("metal");
    const auto invalid = us4::ParseBackendType("cuda");
    ok &= Expect(cpu.has_value() && *cpu == us4::BackendType::kScalarCpu,
                 "cpu alias should parse");
    ok &= Expect(metal.has_value() && *metal == us4::BackendType::kMetal,
                 "metal alias should parse");
    ok &= Expect(!invalid.has_value(), "invalid backend should not parse");

    us4::HardwareProbeResult probe = MakeProbe();
    probe.hasMetal = true;
    probe.hasMlx = true;
    probe.hasNeon = true;
    probe.neonVectorBits = 128;
    probe.hasPerformanceCores = true;
    probe.hasEfficiencyCores = true;
    const us4::LlamaAdapter llama;
    const us4::BackendSelection autoSelection =
        us4::SelectBackend(probe, us4::RuntimeMode::kBalancedPlus, llama);
    ok &= Expect(autoSelection.selected == us4::BackendType::kMetal,
                 "auto backend should prefer metal");
    ok &= Expect(autoSelection.reason == "auto-metal",
                 "auto metal reason should stay explicit");

    const us4::BackendSelection degradedSelection =
        us4::SelectBackend(probe, us4::RuntimeMode::kDegraded, llama);
    ok &= Expect(degradedSelection.selected == us4::BackendType::kMlx,
                 "degraded should prefer mlx before neon");
    ok &= Expect(degradedSelection.reason == "auto-mlx",
                 "auto mlx reason should stay explicit");

    const us4::QwenAdapter qwen;
    const us4::BackendSelection fallbackSelection =
        us4::SelectBackend(MakeProbe(), us4::RuntimeMode::kDegraded, qwen,
                           us4::BackendType::kMetal);
    ok &= Expect(fallbackSelection.selected == us4::BackendType::kNeon,
                 "fallback should pick neon on arm64");
    ok &= Expect(fallbackSelection.fellBack,
                 "requested unavailable backend should mark fallback");

    probe.chip = "Apple M5";
    probe.hasAne = true;
    probe.supportsCoreMl = true;
    const us4::BackendSelection aneSelection =
        us4::SelectBackend(probe, us4::RuntimeMode::kFull, llama);
    ok &= Expect(aneSelection.selected == us4::BackendType::kAne,
                 "m5 full mode should prefer ane");
    ok &= Expect(aneSelection.reason == "auto-ane",
                 "ane auto-selection reason should stay explicit");
  }

  {
    us4::ModelAsset asset;
    std::string error;
    const auto manifest = RepoRoot() / "tests" / "fixtures" / "models" /
                          "qwen-0.5b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(manifest, asset, &error),
                 "qwen manifest should load");
    ok &=
        Expect(asset.family == "qwen", "qwen manifest should preserve family");
    ok &= Expect(asset.modelName == "qwen-0.5b-fixture",
                 "qwen manifest should preserve model name");
  }

  {
    us4::ModelAsset asset;
    std::string error;
    const auto manifest = RepoRoot() / "tests" / "fixtures" / "models" /
                          "llama-3.1-8b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(manifest, asset, &error),
                 "llama manifest should load");
    const us4::LlamaConfig config = us4::ResolveLlamaConfig(&asset);
    ok &= Expect(asset.metadata.contains("rope_theta"),
                 "llama manifest should expose rope theta metadata");
    ok &= Expect(asset.metadata.contains("rope_scale"),
                 "llama manifest should expose rope scale metadata");
    ok &= Expect(!asset.draftModelPath.empty() &&
                     asset.draftModelPath.filename() == "draft-llama.gguf",
                 "llama manifest should surface draft model path");
    ok &= Expect(
        asset.draftModelFormat == us4::ModelFormat::kGguf &&
            asset.sharedTokenizer,
        "llama manifest should surface draft format and shared tokenizer");
    ok &= Expect(config.hiddenSize == 8U && config.queryHeads == 2U &&
                     config.kvHeads == 1U && config.headDim == 4U,
                 "llama config should resolve head topology from manifest");
    ok &= Expect(std::abs(config.ropeTheta - 10000.0F) <= 1e-6F,
                 "llama config should preserve rope theta");
    ok &= Expect(config.ropeScaling == us4::RopeScalingType::kDynamic,
                 "llama config should preserve rope scaling mode");
    ok &= Expect(std::abs(config.ropeScale - 1.0F) <= 1e-6F,
                 "llama config should preserve rope scale");

    us4::ModelAsset invalidConfigAsset;
    invalidConfigAsset.metadata = {
        {"hidden_size", "10"}, {"query_heads", "0"},   {"kv_heads", "7"},
        {"head_dim", "0"},     {"rope_theta", "0.25"}, {"rope_scaling", "YaRn"},
        {"rope_scale", "-4"},
    };
    const us4::LlamaConfig normalized =
        us4::ResolveLlamaConfig(&invalidConfigAsset);
    ok &= Expect(normalized.hiddenSize == 10U && normalized.queryHeads == 2U &&
                     normalized.kvHeads == 1U && normalized.headDim == 5U,
                 "llama config should normalize invalid head metadata");
    ok &= Expect(normalized.ropeScaling == us4::RopeScalingType::kYaRN,
                 "llama config should parse rope scaling case-insensitively");
    ok &= Expect(std::abs(normalized.ropeTheta - 10000.0F) <= 1e-6F &&
                     std::abs(normalized.ropeScale - 1.0F) <= 1e-6F,
                 "llama config should clamp invalid rope values to defaults");
  }

  {
    const std::array<std::filesystem::path, 4> kMoeAssets = {
        RepoRoot() / "tests" / "fixtures" / "models" / "deepseek-v2-lite" /
            "toy-deepseek.safetensors",
        RepoRoot() / "tests" / "fixtures" / "models" / "glm-5.1" /
            "toy-glm.safetensors",
        RepoRoot() / "tests" / "fixtures" / "models" / "kimi-k2-instruct" /
            "toy-kimi.safetensors",
        RepoRoot() / "tests" / "fixtures" / "models" / "minimax-m2" /
            "toy-minimax.safetensors",
    };
    for (const std::filesystem::path &inputPath : kMoeAssets) {
      us4::ModelAsset asset;
      std::string error;
      ok &=
          Expect(us4::LoadModelAsset(inputPath, asset, &error),
                 "moe binary assets should inherit sibling manifest metadata");
      ok &= Expect(asset.moeLazyLoad,
                   "moe loader contract should preserve lazy-load flag");
      ok &= Expect(asset.moeActiveExperts == 2U,
                   "moe loader contract should preserve active expert count");
      ok &= Expect(asset.expertShardPaths.size() == 2U,
                   "moe loader contract should preserve expert shard list");
    }
  }

  {
    us4::Tensor scalarQuery({1, 2}, us4::DType::kFloat32);
    us4::Tensor scalarKey({2, 2}, us4::DType::kFloat32);
    us4::Tensor scalarValue({2, 2}, us4::DType::kFloat32);
    us4::Tensor scalarOut({1, 2}, us4::DType::kFloat32);
    us4::Tensor gqaSingleOut({1, 2}, us4::DType::kFloat32);
    FillFloatTensor(scalarQuery, {1.0F, 0.0F});
    FillFloatTensor(scalarKey, {1.0F, 0.0F, 0.0F, 1.0F});
    FillFloatTensor(scalarValue, {2.0F, 0.0F, 0.0F, 4.0F});
    std::string attentionError;
    ok &= Expect(us4::ScalarAttention(scalarQuery, scalarKey, scalarValue,
                                      scalarOut, false, {}, &attentionError),
                 "scalar attention reference should succeed");
    ok &= Expect(us4::GqaAttention(scalarQuery, scalarKey, scalarValue, 1U, 1U,
                                   gqaSingleOut, &attentionError),
                 "single-head gqa should succeed");
    const float *scalarOutData = scalarOut.DataAsFloat32();
    const float *gqaSingleData = gqaSingleOut.DataAsFloat32();
    ok &= Expect(std::abs(scalarOutData[0] - gqaSingleData[0]) <= 1e-5F &&
                     std::abs(scalarOutData[1] - gqaSingleData[1]) <= 1e-5F,
                 "single-head gqa should match scalar attention");

    us4::Tensor groupedQuery({1, 4}, us4::DType::kFloat32);
    us4::Tensor groupedKey({2, 2}, us4::DType::kFloat32);
    us4::Tensor groupedValue({2, 2}, us4::DType::kFloat32);
    us4::Tensor groupedOut({1, 4}, us4::DType::kFloat32);
    FillFloatTensor(groupedQuery, {1.0F, 0.0F, 0.0F, 1.0F});
    FillFloatTensor(groupedKey, {1.0F, 0.0F, 0.0F, 1.0F});
    FillFloatTensor(groupedValue, {10.0F, 0.0F, 0.0F, 20.0F});
    ok &= Expect(us4::GqaAttention(groupedQuery, groupedKey, groupedValue, 2U,
                                   1U, groupedOut, &attentionError),
                 "grouped-query attention should succeed");
    const float *groupedData = groupedOut.DataAsFloat32();
    ok &= Expect(std::abs(groupedData[0] - 6.69762F) <= 1e-4F &&
                     std::abs(groupedData[1] - 6.60478F) <= 1e-4F &&
                     std::abs(groupedData[2] - 3.30238F) <= 1e-4F &&
                     std::abs(groupedData[3] - 13.3952F) <= 1e-4F,
                 "grouped-query attention should keep golden output stable");
    ok &=
        Expect(!us4::GqaAttention(groupedQuery, groupedKey, groupedValue, 3U,
                                  2U, groupedOut, &attentionError) &&
                   attentionError == "invalid GQA head relationship",
               "invalid grouped-query topology should fail deterministically");
  }

  {
    us4::UnifiedAllocator allocator;
    allocator.Allocate(32, false);
    const auto shared = allocator.Allocate(128, true);
    ok &= Expect(allocator.SharedAllocationCount() == 1U,
                 "allocator should count unified-shared allocations");
    ok &=
        Expect(shared->visibility == us4::AllocationVisibility::kUnifiedShared,
               "shared allocation should be tagged unified-shared");

    us4::HardwareProbeResult probe = MakeProbe();
    probe.hasMetal = true;
    probe.hasMlx = true;
    us4::RuntimeContext context(probe);
    ok &= Expect(context.metalQueue().Available(),
                 "metal queue should be available on metal probe");
    ok &= Expect(context.metalQueue().Profile().stage ==
                     us4::MetalInitializationStage::kQueueReady,
                 "metal queue should surface queue-ready init stage");
    ok &= Expect(context.metalQueue().Profile().queueCreated,
                 "metal queue should mark queue creation");
    ok &= Expect(
        context.metalQueue().Profile().requiresAutoreleaseBoundary,
        "metal queue should request autorelease boundary on macos probe");
    ok &=
        Expect(context.metalQueue().Device().queueLabel == "us4.metal.default",
               "metal queue should expose queue label");
    ok &= Expect(context.metalQueue().Device().supportsUnifiedMemory,
                 "metal queue should expose unified memory support");
    ok &= Expect(context.mlxBridge().Available(),
                 "mlx bridge should be available on mlx probe");
    ok &= Expect(context.metalQueue().Dispatch(us4::MetalKernelKind::kSoftmax,
                                               2, 64, shared),
                 "metal queue should record dispatch contract");
    ok &= Expect(context.metalQueue().DispatchCount() == 1U,
                 "metal queue should count dispatches");
    ok &= Expect(context.metalQueue().Dispatches().front().entryPoint ==
                     "us4_softmax_rows",
                 "metal queue should surface dispatch entry point");
    ok &= Expect(context.metalQueue().Dispatches().front().relativePath ==
                     "runtime/metal/kernels/softmax.metal",
                 "metal queue should surface dispatch kernel path");
    ok &= Expect(
        context.metalQueue().Dispatches().front().autoreleaseBoundaryRequested,
        "metal queue should record autorelease boundary request");
    const us4::DenseMetalDispatchPlan densePlan =
        us4::BuildDenseMetalDispatchPlan(8, 8, 16);
    ok &= Expect(densePlan.steps.size() == 3U,
                 "dense metal dispatch plan should keep 3 stages");
    const us4::MlxDensePlan mlxPlan = us4::BuildMlxDensePlan(8, 8, 16);
    ok &= Expect(mlxPlan.operations.size() == 3U,
                 "mlx dense plan should keep 3 operations");
    ok &= Expect(context.mlxBridge().BuildDensePlan("llama", 32, shared),
                 "mlx bridge should build dense plan");
    ok &= Expect(context.mlxBridge().EvaluateLastPlan(),
                 "mlx bridge should evaluate last plan");
    ok &= Expect(context.mlxBridge().LastPlan().has_value() &&
                     context.mlxBridge().LastPlan()->operations.size() == 3U,
                 "mlx bridge should surface 3 recorded operations");
    ok &= Expect(us4::GetMetalKernelCatalog().size() == 3U,
                 "metal kernel catalog should keep 3 kernels");
    ok &=
        Expect(us4::FindMetalKernel(us4::MetalKernelKind::kRmsNorm) != nullptr,
               "metal kernel catalog should resolve rmsnorm");
    const us4::ScopedAutoreleasePool pool(true);
    ok &= Expect(pool.Requested(),
                 "autorelease scope should record request intent");
    if (pool.Kind() == us4::AutoreleaseBoundaryKind::kObjectiveC) {
      ok &= Expect(
          pool.Active(),
          "objective-c autorelease scope should be active on apple hosts");
    } else {
      ok &= Expect(
          !pool.Active(),
          "noop autorelease scope should remain inactive off apple hosts");
    }

    us4::HardwareProbeResult aneProbe = probe;
    aneProbe.chip = "Apple M5";
    aneProbe.hasAne = true;
    aneProbe.supportsCoreMl = true;
    aneProbe.recommendedMode = us4::RuntimeMode::kFull;
    us4::RuntimeContext aneContext(aneProbe);
    ok &= Expect(aneContext.aneBackend().Available(),
                 "ane backend should be available on m5 probe");
    ok &= Expect(aneContext.aneBackend().Compile(
                     {.kind = us4::AneModelKind::kAttentionMlp,
                      .family = "llama",
                      .layerName = "decoder.block.0.mlp",
                      .tokenCount = 8U,
                      .usesSharedTokenizer = true,
                      .staticShapePreferred = true}),
                 "ane backend should record compile intent");
    ok &= Expect(
        aneContext.aneBackend().LastCompiledModel().has_value() &&
            aneContext.aneBackend().LastCompiledModel()->supportsPrediction,
        "ane backend should keep compiled model prediction intent");
    ok &= Expect(aneContext.aneBackend().Predict(3U, 1U),
                 "ane backend should record predict intent");
    ok &= Expect(aneContext.aneBackend().PredictionCount() == 1U,
                 "ane backend should count predictions");
    const us4::OffloadDecision eligibleLayer =
        aneContext.layerOffloader().Decide(
            {.family = "llama",
             .layerName = "decoder.block.0.mlp.up",
             .layerType = us4::OffloadLayerType::kMlpUpProjection,
             .mode = us4::RuntimeMode::kFull,
             .tokenCount = 8U,
             .hiddenSize = 4096U,
             .weightDType = us4::DType::kFloat16,
             .staticShape = true});
    ok &= Expect(eligibleLayer.eligible && !eligibleLayer.fallbackToMetal &&
                     eligibleLayer.backend == "ane",
                 "ane layer offloader should accept static mlp projections");
    const us4::OffloadDecision fallbackLayer =
        aneContext.layerOffloader().Decide(
            {.family = "deepseek",
             .layerName = "router.0",
             .layerType = us4::OffloadLayerType::kRouter,
             .mode = us4::RuntimeMode::kFull,
             .tokenCount = 8U,
             .hiddenSize = 4096U,
             .weightDType = us4::DType::kFloat16,
             .staticShape = true});
    ok &= Expect(!fallbackLayer.eligible && fallbackLayer.fallbackToMetal &&
                     fallbackLayer.reason == "ane-router-cpu",
                 "ane layer offloader should keep router layers off ANE");
    const us4::MixedDispatchPlan mixedPlan =
        aneContext.mixedDispatch().BuildPlan(
            "llama", 16U, 4096U, us4::DType::kFloat16, us4::RuntimeMode::kFull);
    ok &= Expect(aneContext.mixedDispatch().Available(),
                 "mixed dispatch should be available on m5 probe");
    ok &= Expect(mixedPlan.stages.size() == 3U,
                 "mixed dispatch plan should keep 3 canonical stages");
    ok &= Expect(
        mixedPlan.stages.front().backend == us4::DispatchBackend::kAne,
        "mixed dispatch should prefer ane for eligible full-mode stages");
    const auto aneShared = aneContext.allocator().Allocate(2048U, true);
    const us4::MixedDispatchTelemetry mixedTelemetry =
        aneContext.mixedDispatch().Execute(
            mixedPlan, aneContext.layerOffloader(), aneContext.aneBackend(),
            aneContext.metalQueue(), aneShared);
    ok &= Expect(mixedTelemetry.aneStages == 3U &&
                     mixedTelemetry.metalStages == 0U &&
                     mixedTelemetry.strategy == "ane-only",
                 "mixed dispatch should execute fully on ane when all stages "
                 "are eligible");
    const us4::MixedDispatchPlan fallbackPlan =
        aneContext.mixedDispatch().BuildPlan(
            "llama", 16U, 4096U, us4::DType::kInt8, us4::RuntimeMode::kFull);
    const us4::MixedDispatchTelemetry fallbackTelemetry =
        aneContext.mixedDispatch().Execute(
            fallbackPlan, aneContext.layerOffloader(), aneContext.aneBackend(),
            aneContext.metalQueue(), aneShared);
    ok &= Expect(fallbackTelemetry.metalStages == 3U &&
                     fallbackTelemetry.aneStages == 0U &&
                     fallbackTelemetry.strategy == "metal-only",
                 "mixed dispatch should fall back to metal for low-bit plans");

    us4::HardwareProbeResult thermalProbe = aneProbe;
    thermalProbe.unifiedMemoryGiB = 24ULL;
    thermalProbe.recommendedMode = us4::RuntimeMode::kFull;
    us4::RuntimeContext thermalContext(thermalProbe);
    ok &= Expect(thermalContext.thermalMonitor().Available(),
                 "thermal monitor should be available on apple-like probes");
    ok &= Expect(thermalContext.thermalMonitor().Sample().level ==
                     us4::ThermalPressureLevel::kElevated,
                 "thermal monitor should surface elevated pressure");
    ok &= Expect(thermalContext.mode() == us4::RuntimeMode::kBalancedPlus,
                 "thermal monitor should downgrade full mode under elevated "
                 "pressure");
    thermalContext.SetMode(us4::RuntimeMode::kDegraded);
    ok &= Expect(!thermalContext.thermalMonitor().LastDecision().downgraded &&
                     thermalContext.mode() == us4::RuntimeMode::kDegraded,
                 "thermal monitor should leave already lower modes intact");
  }

  {
    us4::Tensor linear({1, 4}, us4::DType::kFloat32);
    us4::Tensor dynamic({1, 4}, us4::DType::kFloat32);
    us4::Tensor yarn({1, 4}, us4::DType::kFloat32);
    float *linearData = linear.MutableDataAsFloat32();
    float *dynamicData = dynamic.MutableDataAsFloat32();
    float *yarnData = yarn.MutableDataAsFloat32();
    linearData[0] = dynamicData[0] = yarnData[0] = 1.0F;
    linearData[1] = dynamicData[1] = yarnData[1] = 0.0F;
    linearData[2] = dynamicData[2] = yarnData[2] = 0.5F;
    linearData[3] = dynamicData[3] = yarnData[3] = 0.25F;

    us4::ApplyRopeInPlace(linear, 256U, 10000.0F, us4::RopeScalingType::kLinear,
                          4.0F);
    us4::ApplyRopeInPlace(dynamic, 256U, 10000.0F,
                          us4::RopeScalingType::kDynamic, 4.0F);
    us4::ApplyRopeInPlace(yarn, 256U, 10000.0F, us4::RopeScalingType::kYaRN,
                          4.0F);

    const float linearNorm0 = std::sqrt(linearData[0] * linearData[0] +
                                        linearData[1] * linearData[1]);
    const float dynamicNorm0 = std::sqrt(dynamicData[0] * dynamicData[0] +
                                         dynamicData[1] * dynamicData[1]);
    const float yarnNorm2 =
        std::sqrt(yarnData[2] * yarnData[2] + yarnData[3] * yarnData[3]);
    ok &= Expect(std::abs(linearNorm0 - 1.0F) <= 1e-5F,
                 "rope linear scaling should preserve first pair norm");
    ok &= Expect(std::abs(dynamicNorm0 - 1.0F) <= 1e-5F,
                 "rope dynamic scaling should preserve first pair norm");
    ok &= Expect(std::abs(yarnNorm2 - std::sqrt(0.3125F)) <= 1e-5F,
                 "rope yarn scaling should preserve second pair norm");
    ok &= Expect(std::abs(linearData[0] - dynamicData[0]) > 1e-5F,
                 "rope dynamic scaling should differ from linear output");
    ok &= Expect(std::abs(dynamicData[0] - yarnData[0]) > 1e-5F,
                 "rope yarn scaling should differ from dynamic output");

    us4::Tensor clamped({1, 4}, us4::DType::kFloat32);
    float *clampedData = clamped.MutableDataAsFloat32();
    clampedData[0] = 1.0F;
    clampedData[1] = 0.0F;
    clampedData[2] = 0.5F;
    clampedData[3] = 0.25F;
    us4::ApplyRopeInPlace(clamped, 8U, 0.25F, us4::RopeScalingType::kLinear,
                          -2.0F);
    us4::Tensor defaults({1, 4}, us4::DType::kFloat32);
    float *defaultData = defaults.MutableDataAsFloat32();
    defaultData[0] = 1.0F;
    defaultData[1] = 0.0F;
    defaultData[2] = 0.5F;
    defaultData[3] = 0.25F;
    us4::ApplyRopeInPlace(defaults, 8U, 10000.0F, us4::RopeScalingType::kLinear,
                          1.0F);
    ok &= Expect(std::abs(clampedData[0] - defaultData[0]) <= 1e-5F &&
                     std::abs(clampedData[2] - defaultData[2]) <= 1e-5F,
                 "rope should clamp invalid theta and scale to safe defaults");
  }

  {
    us4::PrefixCache cache;
    cache.Retain("hello");
    cache.Retain("hello");
    const auto retained = cache.Lookup("hello");
    ok &= Expect(retained.has_value() && retained->refCount == 2U,
                 "prefix cache should count retains");
    cache.Release("hello");
    cache.Release("hello");
    ok &= Expect(!cache.Lookup("hello").has_value(),
                 "prefix cache should erase on last release");

    us4::KvPager pager(1);
    pager.Append("prompt-a", {1.0F, 2.0F});
    pager.Append("prompt-b", {3.0F, 4.0F});
    const auto page = pager.Lookup("prompt-a");
    ok &= Expect(page.has_value(), "kv pager should find stored page");
    ok &= Expect(page->keys.size() == 2U,
                 "kv pager should preserve key rows for cached pages");
    ok &= Expect(page->hitCount >= 1U,
                 "kv pager lookup should increase hit count");
    ok &= Expect(pager.WarmPageCount() == 1U,
                 "kv pager should demote excess hot pages into warm tier");

    us4::Summarizer summarizer;
    const auto summary = summarizer.Summarize({2.0F, 4.0F, 6.0F});
    ok &= Expect(summary.size() == 1U && summary[0] == 4.0F,
                 "summarizer should produce arithmetic mean");
    const auto rowSummary =
        summarizer.SummarizeRows({1.0F, 3.0F, 5.0F, 7.0F, 9.0F, 11.0F}, 3U);
    ok &= Expect(rowSummary.size() == 3U && rowSummary[0] == 4.0F &&
                     rowSummary[1] == 6.0F && rowSummary[2] == 8.0F,
                 "summarizer should compress rows per hidden column");
  }

  {
    us4::Router router;
    const us4::RouterDecision decision =
        router.RouteTopK({0.1F, 0.8F, 0.4F, 0.7F}, 2);
    ok &= Expect(decision.selected.size() == 2U,
                 "router should clamp top-k to requested size");
    ok &= Expect(decision.selected[0].expert == 1U,
                 "router should sort highest score first");
    ok &= Expect(decision.selected[0].logit == 0.8F,
                 "router should preserve raw logit in telemetry");
    ok &= Expect(decision.entropy > 0.0F,
                 "router should expose positive routing entropy");
    ok &= Expect(decision.loadBalance > 0.0F && decision.loadBalance <= 1.0F,
                 "router should expose normalized load balance");
    ok &= Expect(decision.selectedMass > 0.0F && decision.selectedMass <= 1.0F,
                 "router should expose selected probability mass");
    ok &= Expect(router.LastDecision().has_value() &&
                     router.LastDecision()->totalExperts == 4U,
                 "router should retain the last routing decision");

    us4::ExpertPager pager(2);
    pager.Touch("expert-a");
    pager.Touch("expert-b");
    pager.Touch("expert-a");
    pager.Touch("expert-c");
    ok &= Expect(pager.ResidentCount() == 2U,
                 "expert pager should enforce resident limit");
    ok &= Expect(pager.IsResident("expert-a"),
                 "expert pager should keep hot expert resident");
    ok &= Expect(pager.LoadCount() == 3U,
                 "expert pager should count visible loads");
    ok &= Expect(pager.ReuseCount() == 1U,
                 "expert pager should count visible reuse");
    ok &= Expect(pager.EvictionCount() == 1U,
                 "expert pager should count visible evictions");
    const us4::ExpertPagerSnapshot pagerSnapshot = pager.Snapshot();
    ok &= Expect(pagerSnapshot.residentCount == 2U &&
                     pagerSnapshot.residents.size() == 2U,
                 "expert pager snapshot should expose resident state");
  }

  {
    const us4::IUS4V6Adapter *qwen = us4::FindAdapterByModel("QWEN-0.5B");
    const us4::IUS4V6Adapter *llama = us4::FindAdapterByModel("llama-3.1-8b");
    const us4::IUS4V6Adapter *bitnet =
        us4::FindAdapterByModel("bitnet-b1.58-2b");
    const us4::IUS4V6Adapter *ternaryAdapter =
        us4::FindAdapterByModel("pt-bitnet-ternary-2b");
    const us4::IUS4V6Adapter *ternary =
        us4::FindAdapterByModel("pt-bitnet-ternary-2b");
    ok &= Expect(qwen != nullptr, "registry should find qwen by model");
    ok &= Expect(llama != nullptr, "registry should find llama by model");
    ok &= Expect(bitnet != nullptr, "registry should find bitnet by model");
    ok &= Expect(ternaryAdapter != nullptr,
                 "registry should find ternary adapter by model");
    ok &= Expect(ternary != nullptr,
                 "registry should find ternary by exact model");
    ok &= Expect(ternary != nullptr && ternary->Family() == "ternary",
                 "registry should not misroute ternary to bitnet");

    us4::ModelAsset asset;
    std::string error;
    const auto manifest = RepoRoot() / "tests" / "fixtures" / "models" /
                          "qwen-0.5b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(manifest, asset, &error),
                 "adapter generation manifest should load");
    us4::RuntimeContext context(MakeProbe());
    qwen->ConfigureRuntime(context);
    const us4::GenerationResult result =
        qwen->Generate({.prompt = "Hi, US4!",
                        .maxTokens = 4,
                        .asset = &asset,
                        .requestedBackend = us4::BackendType::kMetal},
                       context);
    ok &= Expect(result.modelName == "qwen-0.5b-fixture",
                 "generation should surface manifest model name");
    ok &= Expect(result.backendReason == "requested-backend-unavailable",
                 "generation should expose fallback reason");
    ok &= Expect(result.fellBack, "generation should mark backend fallback");
    ok &= Expect(result.sharedAllocations == 0U,
                 "scalar fallback should not record shared allocations");
    ok &= Expect(result.metalDispatches == 0U,
                 "scalar fallback should not record metal dispatches");
    ok &= Expect(result.mlxOperationCount == 0U,
                 "scalar fallback should not record mlx operations");
    ok &= Expect(!result.kvCacheHit,
                 "first generation should populate kv cache on demand");
    ok &= Expect(result.kvPageCount == 1U,
                 "first generation should publish a single prompt kv page");
    ok &= Expect(result.kvHotPages == 1U && result.kvWarmPages == 0U &&
                     result.kvColdPages == 0U,
                 "single prompt cache should stay hot");
    ok &= Expect(result.prefixCacheEntries == 1U,
                 "generation should retain a prefix cache entry");

    const us4::GenerationResult autoResult = qwen->Generate(
        {.prompt = "Hi, US4!", .maxTokens = 4, .asset = &asset}, context);
    ok &= Expect(autoResult.backendReason == "auto-neon" ||
                     autoResult.backendReason == "auto-scalar",
                 "auto generation should expose explicit backend reason");
    ok &= Expect(autoResult.kvCacheHit,
                 "repeated generation should reuse prompt kv cache");
    ok &= Expect(!autoResult.kvRestoredFromColdStore,
                 "shared runtime context should hit hot kv before cold store");
    ok &= Expect(result.weightDType == "fp16",
                 "generation should surface asset weight dtype");
    ok &= Expect(result.neonKernelFlavor == "fp16-lane8",
                 "generation should surface neon flavor for fp16 assets");
    ok &= Expect(result.dequantPath == "none",
                 "fp16 assets should not request dequant path");

    std::filesystem::remove_all(RepoRoot() / "build" / "kv-cold-store");
    us4::RuntimeContext lowMemoryContext(MakeProbe());
    lowMemoryContext.SetMode(us4::RuntimeMode::kMicro);
    const us4::GenerationResult summarizedResult =
        qwen->Generate({.prompt = "one two three four five six seven",
                        .maxTokens = 2,
                        .asset = &asset},
                       lowMemoryContext);
    ok &= Expect(summarizedResult.kvSummaryRows > 0U,
                 "micro mode should summarize old kv rows");

    us4::RuntimeContext restoredContext(MakeProbe());
    restoredContext.SetMode(us4::RuntimeMode::kMicro);
    const us4::GenerationResult restoredResult =
        qwen->Generate({.prompt = "one two three four five six seven",
                        .maxTokens = 2,
                        .asset = &asset},
                       restoredContext);
    ok &= Expect(restoredResult.kvCacheHit,
                 "second low-memory pass should reuse kv state");
    ok &= Expect(restoredResult.kvRestoredFromColdStore,
                 "fresh low-memory context should restore kv from cold store");

    us4::HardwareProbeResult appleProbe = MakeProbe();
    appleProbe.hasMetal = true;
    appleProbe.hasMlx = true;
    appleProbe.unifiedMemoryGiB = 96;
    appleProbe.recommendedMode = us4::RuntimeMode::kBalancedPlus;
    us4::RuntimeContext acceleratedContext(appleProbe);
    const us4::GenerationResult llamaResult =
        llama->Generate({.prompt = "metal path",
                         .maxTokens = 4,
                         .requestedBackend = us4::BackendType::kMetal},
                        acceleratedContext);
    ok &= Expect(llamaResult.backend == "metal",
                 "llama should keep metal backend when available");
    ok &= Expect(acceleratedContext.metalQueue().DispatchCount() == 3U,
                 "llama generation should touch the metal scaffold");
    ok &= Expect(acceleratedContext.allocator().SharedAllocationCount() == 1U,
                 "metal scaffold should allocate unified-shared memory");
    ok &= Expect(llamaResult.sharedAllocations == 1U,
                 "result should surface shared allocation count");
    ok &= Expect(llamaResult.metalDispatches == 3U,
                 "result should surface metal dispatch count");
    ok &= Expect(llamaResult.mlxOperationCount == 0U,
                 "metal path should not report mlx operations");
    ok &= Expect(llamaResult.metalQueueLabel == "us4.metal.default",
                 "result should surface metal queue label");
    ok &= Expect(qwen->SupportsMetalBackend(),
                 "qwen should declare metal support");

    us4::ModelAsset bitnetAsset;
    const auto bitnetManifest = RepoRoot() / "tests" / "fixtures" / "models" /
                                "bitnet-b1.58-2b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(bitnetManifest, bitnetAsset, &error),
                 "bitnet manifest should load");
    us4::RuntimeContext bitnetContext(MakeProbe());
    bitnet->ConfigureRuntime(bitnetContext);
    const us4::GenerationResult bitnetResult = bitnet->Generate(
        {.prompt = "hi", .maxTokens = 2, .asset = &bitnetAsset}, bitnetContext);
    ok &= Expect(bitnetResult.weightDType == "int8",
                 "bitnet generation should surface int8 weight dtype");
    ok &= Expect(bitnetResult.dequantPath == "groupwise-int8",
                 "bitnet generation should surface int8 dequant path");
    ok &= Expect(bitnetResult.neonKernelFlavor == "int8-dot",
                 "bitnet generation should surface int8 dot neon flavor");

    us4::ModelAsset ternaryAsset;
    const auto ternaryManifest = RepoRoot() / "tests" / "fixtures" / "models" /
                                 "pt-bitnet-ternary-2b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(ternaryManifest, ternaryAsset, &error),
                 "ternary manifest should load");
    us4::RuntimeContext ternaryContext(MakeProbe());
    ternaryAdapter->ConfigureRuntime(ternaryContext);
    const us4::GenerationResult ternaryResult = ternaryAdapter->Generate(
        {.prompt = "hi", .maxTokens = 2, .asset = &ternaryAsset},
        ternaryContext);
    ok &= Expect(ternaryResult.weightDType == "int4",
                 "ternary generation should surface int4 weight dtype");
    ok &= Expect(ternaryResult.dequantPath == "groupwise-int4",
                 "ternary generation should surface int4 dequant path");
    ok &= Expect(ternaryResult.neonKernelFlavor == "scalar-bridge",
                 "ternary generation should surface scalar-bridge neon flavor");
  }

  {
    const us4::IUS4V6Adapter *deepseek =
        us4::FindAdapterByModel("deepseek-v2-lite");
    ok &= Expect(deepseek != nullptr,
                 "registry should find deepseek moe adapter by model");
    us4::RuntimeContext moeContext(MakeProbe());
    deepseek->ConfigureRuntime(moeContext);
    const us4::GenerationResult deepseekResult =
        deepseek->Generate({.prompt = "hi", .maxTokens = 2}, moeContext);
    ok &= Expect(deepseekResult.family == "deepseek",
                 "deepseek generation should preserve moe family");
    ok &= Expect(deepseekResult.moeSelectedExperts == 2U,
                 "deepseek generation should expose selected expert count");
    ok &= Expect(deepseekResult.moeRouterEntropy > 0.0F,
                 "deepseek generation should expose router entropy");
    ok &= Expect(deepseekResult.moeLoadBalance > 0.0F &&
                     deepseekResult.moeLoadBalance <= 1.0F,
                 "deepseek generation should expose load balance score");
    ok &= Expect(deepseekResult.moeSelectedMass > 0.0F &&
                     deepseekResult.moeSelectedMass <= 1.0F,
                 "deepseek generation should expose selected expert mass");
    ok &= Expect(deepseekResult.moePagerLoads == 2U,
                 "deepseek generation should expose pager loads");
    ok &= Expect(deepseekResult.moePagerEvictions == 0U,
                 "deepseek generation should expose pager evictions");
    ok &= Expect(deepseekResult.moePagerReuses == 0U,
                 "deepseek generation should expose pager reuse");
    ok &= Expect(deepseekResult.moeResidentExperts == 2U,
                 "deepseek generation should expose resident expert count");
    ok &= Expect(deepseekResult.text.find("moe-route") != std::string::npos,
                 "deepseek generation should surface routed expert signature");

    const us4::GenerationResult repeatedDeepseek =
        deepseek->Generate({.prompt = "hi", .maxTokens = 2}, moeContext);
    ok &= Expect(repeatedDeepseek.moePagerLoads == 2U,
                 "deepseek repeat should not add extra loads for same experts");
    ok &= Expect(repeatedDeepseek.moePagerReuses >= 2U,
                 "deepseek repeat should surface pager reuse");
  }

  {
    const us4::IUS4V6Adapter *glm = us4::FindAdapterByModel("glm-5.1");
    ok &= Expect(glm != nullptr, "glm adapter should stay registered");
    if (glm != nullptr) {
      us4::RuntimeContext moeContext(MakeProbe());
      glm->ConfigureRuntime(moeContext);
      const us4::GenerationResult glmResult = glm->Generate(
          {.prompt = "tool reason vision", .maxTokens = 2}, moeContext);
      ok &= Expect(glmResult.family == "glm",
                   "glm generation should preserve moe family");
      ok &= Expect(glmResult.moeSelectedExperts == 2U,
                   "glm generation should expose selected expert count");
      ok &= Expect(glmResult.moeRouterEntropy > 0.0F,
                   "glm generation should expose router entropy");
      ok &= Expect(glmResult.moeLoadBalance > 0.0F &&
                       glmResult.moeLoadBalance <= 1.0F,
                   "glm generation should expose load balance score");
      ok &= Expect(glmResult.moeSelectedMass > 0.0F &&
                       glmResult.moeSelectedMass <= 1.0F,
                   "glm generation should expose selected expert mass");
      ok &= Expect(glmResult.moePagerLoads == 2U,
                   "glm generation should expose pager loads");
      ok &= Expect(glmResult.moePagerEvictions == 0U,
                   "glm generation should expose pager evictions");
      ok &= Expect(glmResult.moePagerReuses == 0U,
                   "glm generation should expose pager reuse");
      ok &= Expect(glmResult.moeResidentExperts == 2U,
                   "glm generation should expose resident expert count");
      ok &= Expect(glmResult.text.find("glm-route") != std::string::npos,
                   "glm generation should surface routed expert signature");

      const us4::GenerationResult repeatedGlm = glm->Generate(
          {.prompt = "tool reason vision", .maxTokens = 2}, moeContext);
      ok &= Expect(repeatedGlm.moePagerLoads == 2U,
                   "glm repeat should not add extra loads for same experts");
      ok &= Expect(repeatedGlm.moePagerReuses >= 2U,
                   "glm repeat should surface pager reuse");
    }
  }

  {
    const us4::IUS4V6Adapter *kimi =
        us4::FindAdapterByModel("kimi-k2-instruct");
    ok &= Expect(kimi != nullptr, "kimi adapter should stay registered");
    if (kimi != nullptr) {
      us4::RuntimeContext moeContext(MakeProbe());
      kimi->ConfigureRuntime(moeContext);
      const us4::GenerationResult kimiResult = kimi->Generate(
          {.prompt = "smart context", .maxTokens = 2}, moeContext);
      ok &= Expect(kimiResult.family == "kimi",
                   "kimi generation should preserve moe family");
      ok &= Expect(kimiResult.moeSelectedExperts == 2U,
                   "kimi generation should expose selected expert count");
      ok &= Expect(kimiResult.moeRouterEntropy > 0.0F,
                   "kimi generation should expose router entropy");
      ok &= Expect(kimiResult.moeLoadBalance > 0.0F &&
                       kimiResult.moeLoadBalance <= 1.0F,
                   "kimi generation should expose load balance score");
      ok &= Expect(kimiResult.moeSelectedMass > 0.0F &&
                       kimiResult.moeSelectedMass <= 1.0F,
                   "kimi generation should expose selected expert mass");
      ok &= Expect(kimiResult.moePagerLoads == 2U,
                   "kimi generation should expose pager loads");
      ok &= Expect(kimiResult.moePagerEvictions == 0U,
                   "kimi generation should expose pager evictions");
      ok &= Expect(kimiResult.moePagerReuses == 0U,
                   "kimi generation should expose pager reuse");
      ok &= Expect(kimiResult.moeResidentExperts == 2U,
                   "kimi generation should expose resident expert count");
      ok &= Expect(kimiResult.text.find("kimi-route") != std::string::npos,
                   "kimi generation should surface routed expert signature");

      const us4::GenerationResult repeatedKimi = kimi->Generate(
          {.prompt = "smart context", .maxTokens = 2}, moeContext);
      ok &= Expect(repeatedKimi.moePagerLoads == 2U,
                   "kimi repeat should not add extra loads for same experts");
      ok &= Expect(repeatedKimi.moePagerReuses >= 2U,
                   "kimi repeat should surface pager reuse");
    }
  }

  {
    const us4::IUS4V6Adapter *minimax = us4::FindAdapterByModel("minimax-m2");
    ok &= Expect(minimax != nullptr, "minimax adapter should stay registered");
    if (minimax != nullptr) {
      us4::RuntimeContext moeContext(MakeProbe());
      minimax->ConfigureRuntime(moeContext);
      const us4::GenerationResult minimaxResult = minimax->Generate(
          {.prompt = "image audio fusion", .maxTokens = 2}, moeContext);
      ok &= Expect(minimaxResult.family == "minimax",
                   "minimax generation should preserve moe family");
      ok &= Expect(minimaxResult.moeSelectedExperts == 2U,
                   "minimax generation should expose selected expert count");
      ok &= Expect(minimaxResult.moeRouterEntropy > 0.0F,
                   "minimax generation should expose router entropy");
      ok &= Expect(minimaxResult.moeLoadBalance > 0.0F &&
                       minimaxResult.moeLoadBalance <= 1.0F,
                   "minimax generation should expose load balance score");
      ok &= Expect(minimaxResult.moeSelectedMass > 0.0F &&
                       minimaxResult.moeSelectedMass <= 1.0F,
                   "minimax generation should expose selected expert mass");
      ok &= Expect(minimaxResult.moePagerLoads == 2U,
                   "minimax generation should expose pager loads");
      ok &= Expect(minimaxResult.moePagerEvictions == 0U,
                   "minimax generation should expose pager evictions");
      ok &= Expect(minimaxResult.moePagerReuses == 0U,
                   "minimax generation should expose pager reuse");
      ok &= Expect(minimaxResult.moeResidentExperts == 2U,
                   "minimax generation should expose resident expert count");
      ok &=
          Expect(minimaxResult.text.find("minimax-route") != std::string::npos,
                 "minimax generation should surface routed expert signature");

      const us4::GenerationResult repeatedMiniMax = minimax->Generate(
          {.prompt = "image audio fusion", .maxTokens = 2}, moeContext);
      ok &=
          Expect(repeatedMiniMax.moePagerLoads == 2U,
                 "minimax repeat should not add extra loads for same experts");
      ok &= Expect(repeatedMiniMax.moePagerReuses >= 2U,
                   "minimax repeat should surface pager reuse");
    }
  }

  {
    us4::HardwareProbeResult neonProbe = MakeProbe();
    neonProbe.hasNeon = true;
    neonProbe.neonVectorBits = 128;
    neonProbe.hasPerformanceCores = true;
    neonProbe.hasEfficiencyCores = true;

    const us4::Tensor fp16Lhs({8, 16}, us4::DType::kFloat16,
                              us4::DeviceType::kCpu);
    const us4::Tensor fp16Rhs({16, 32}, us4::DType::kFloat16,
                              us4::DeviceType::kCpu);
    const us4::NeonMatmulProfile matmulProfile =
        us4::PlanNeonMatmul(neonProbe, fp16Lhs, fp16Rhs);
    ok &= Expect(matmulProfile.flavor == us4::NeonKernelFlavor::kFp16Lane8,
                 "neon matmul should pick fp16 lane8 profile on arm64");
    ok &= Expect(matmulProfile.tileRows == 8U && matmulProfile.tileCols == 8U,
                 "neon matmul should keep 8x8 tile contract");
    const us4::Tensor bf16Lhs({8, 16}, us4::DType::kBFloat16,
                              us4::DeviceType::kCpu);
    const us4::Tensor bf16Rhs({16, 24}, us4::DType::kBFloat16,
                              us4::DeviceType::kCpu);
    const us4::NeonMatmulProfile bf16MatmulProfile =
        us4::PlanNeonMatmul(neonProbe, bf16Lhs, bf16Rhs);
    ok &= Expect(bf16MatmulProfile.flavor == us4::NeonKernelFlavor::kBf16Lane8,
                 "neon matmul should pick bf16 lane8 profile on arm64");
    ok &= Expect(bf16MatmulProfile.tileRows == 8U &&
                     bf16MatmulProfile.tileCols == 8U,
                 "neon matmul should keep 8x8 tile contract for bf16");

    const us4::Tensor query({1, 8, 64}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    const us4::Tensor key({1, 8, 64}, us4::DType::kFloat32,
                          us4::DeviceType::kCpu);
    const us4::Tensor value({1, 8, 64}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    const us4::NeonAttentionProfile attentionProfile =
        us4::PlanNeonAttention(neonProbe, query, key, value, true);
    ok &=
        Expect(attentionProfile.fusesSoftmaxRescale,
               "neon attention should preserve fused softmax-rescale contract");
    ok &=
        Expect(attentionProfile.headDimBlock == 32U,
               "neon attention should keep 32-wide head blocks when possible");

    us4::Tensor attentionQuery({2, 4}, us4::DType::kFloat32,
                               us4::DeviceType::kCpu);
    us4::Tensor attentionKey({3, 4}, us4::DType::kFloat32,
                             us4::DeviceType::kCpu);
    us4::Tensor attentionValue({3, 3}, us4::DType::kFloat32,
                               us4::DeviceType::kCpu);
    us4::Tensor neonAttentionOut({2, 3}, us4::DType::kFloat32,
                                 us4::DeviceType::kCpu);
    us4::Tensor scalarAttentionOut({2, 3}, us4::DType::kFloat32,
                                   us4::DeviceType::kCpu);
    float *attentionQueryData = attentionQuery.MutableDataAsFloat32();
    float *attentionKeyData = attentionKey.MutableDataAsFloat32();
    float *attentionValueData = attentionValue.MutableDataAsFloat32();
    for (std::size_t index = 0; index < 8U; ++index) {
      attentionQueryData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.25F) - 0.5F;
    }
    for (std::size_t index = 0; index < 12U; ++index) {
      attentionKeyData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.20F) - 0.25F;
    }
    for (std::size_t index = 0; index < 9U; ++index) {
      attentionValueData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.15F) + 0.10F;
    }
    ok &=
        Expect(us4::NeonAttention(attentionQuery, attentionKey, attentionValue,
                                  neonAttentionOut, false, {}, nullptr),
               "neon attention should execute fp32 path");
    ok &= Expect(us4::ScalarAttention(attentionQuery, attentionKey,
                                      attentionValue, scalarAttentionOut, false,
                                      {}, nullptr),
                 "scalar attention should provide reference path");
    const float *neonAttentionValues = neonAttentionOut.DataAsFloat32();
    const float *scalarAttentionValues = scalarAttentionOut.DataAsFloat32();
    bool attentionMatches =
        neonAttentionValues != nullptr && scalarAttentionValues != nullptr;
    for (std::size_t index = 0; attentionMatches && index < 6U; ++index) {
      const float diff =
          neonAttentionValues[index] - scalarAttentionValues[index];
      attentionMatches = std::abs(diff) <= 1e-5F;
    }
    ok &= Expect(attentionMatches,
                 "neon attention should match scalar fp32 outputs");

    us4::Tensor cacheKeys({1, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor cacheValues({1, 3}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    us4::Tensor causalNeonOut({2, 3}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor causalScalarOut({2, 3}, us4::DType::kFloat32,
                                us4::DeviceType::kCpu);
    float *cacheKeyData = cacheKeys.MutableDataAsFloat32();
    float *cacheValueData = cacheValues.MutableDataAsFloat32();
    for (std::size_t index = 0; index < 4U; ++index) {
      cacheKeyData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.07F) + 0.11F;
    }
    for (std::size_t index = 0; index < 3U; ++index) {
      cacheValueData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.18F) + 0.02F;
    }
    const us4::AttentionCacheView cache{&cacheKeys, &cacheValues};
    ok &=
        Expect(us4::NeonAttention(attentionQuery, attentionKey, attentionValue,
                                  causalNeonOut, true, cache, nullptr),
               "neon attention should support causal cache path");
    ok &= Expect(us4::ScalarAttention(attentionQuery, attentionKey,
                                      attentionValue, causalScalarOut, true,
                                      cache, nullptr),
                 "scalar attention should support causal cache reference");
    const float *causalNeonValues = causalNeonOut.DataAsFloat32();
    const float *causalScalarValues = causalScalarOut.DataAsFloat32();
    bool causalMatches =
        causalNeonValues != nullptr && causalScalarValues != nullptr;
    for (std::size_t index = 0; causalMatches && index < 6U; ++index) {
      const float diff = causalNeonValues[index] - causalScalarValues[index];
      causalMatches = std::abs(diff) <= 1e-5F;
    }
    ok &= Expect(causalMatches,
                 "neon attention should match scalar outputs with cache");

    us4::Tensor wideValue({3, 5}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor wideCacheValues({1, 5}, us4::DType::kFloat32,
                                us4::DeviceType::kCpu);
    us4::Tensor wideNeonOut({2, 5}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    us4::Tensor wideScalarOut({2, 5}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    float *wideValueData = wideValue.MutableDataAsFloat32();
    float *wideCacheValueData = wideCacheValues.MutableDataAsFloat32();
    for (std::size_t index = 0; index < 15U; ++index) {
      wideValueData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.11F) - 0.04F;
    }
    for (std::size_t index = 0; index < 5U; ++index) {
      wideCacheValueData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.16F) + 0.07F;
    }
    const us4::AttentionCacheView wideCache{&cacheKeys, &wideCacheValues};
    ok &= Expect(us4::NeonAttention(attentionQuery, attentionKey, wideValue,
                                    wideNeonOut, true, wideCache, nullptr),
                 "neon attention should support wide value tail accumulation");
    ok &= Expect(us4::ScalarAttention(attentionQuery, attentionKey, wideValue,
                                      wideScalarOut, true, wideCache, nullptr),
                 "scalar attention should provide wide value tail reference");
    const float *wideNeonValues = wideNeonOut.DataAsFloat32();
    const float *wideScalarValues = wideScalarOut.DataAsFloat32();
    bool wideMatches = wideNeonValues != nullptr && wideScalarValues != nullptr;
    for (std::size_t index = 0; wideMatches && index < 10U; ++index) {
      const float diff = wideNeonValues[index] - wideScalarValues[index];
      wideMatches = std::abs(diff) <= 1e-5F;
    }
    ok &= Expect(
        wideMatches,
        "neon attention should match scalar outputs for wide value tails");

    us4::Tensor fp16MatmulLhs({2, 3}, us4::DType::kFloat16,
                              us4::DeviceType::kCpu);
    us4::Tensor fp16MatmulRhs({3, 5}, us4::DType::kFloat16,
                              us4::DeviceType::kCpu);
    us4::Tensor fp16NeonOut({2, 5}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    us4::Tensor fp16ScalarLhs({2, 3}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor fp16ScalarRhs({3, 5}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor fp16ScalarOut({2, 5}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    const std::vector<float> fp16LhsValues = {0.5F, -1.0F, 2.0F,
                                              1.5F, 0.25F, -0.75F};
    const std::vector<float> fp16RhsValues = {
        0.25F, -0.5F, 1.0F,   0.75F, -1.25F, 1.5F,  0.0F, -0.25F,
        0.5F,  1.0F,  -0.75F, 0.25F, 0.5F,   -1.5F, 0.25F};
    FillHalfTensor(fp16MatmulLhs, fp16LhsValues, false);
    FillHalfTensor(fp16MatmulRhs, fp16RhsValues, false);
    ok &= Expect(FillHalfReferenceTensor(fp16ScalarLhs, fp16LhsValues, false),
                 "scalar matmul should receive fp16-rounded lhs reference");
    ok &= Expect(FillHalfReferenceTensor(fp16ScalarRhs, fp16RhsValues, false),
                 "scalar matmul should receive fp16-rounded rhs reference");
    ok &= Expect(
        us4::NeonMatmul(fp16MatmulLhs, fp16MatmulRhs, fp16NeonOut, nullptr),
        "neon matmul should execute fp16 inputs");
    ok &= Expect(
        us4::ScalarMatmul(fp16ScalarLhs, fp16ScalarRhs, fp16ScalarOut, nullptr),
        "scalar matmul should provide fp16 reference");
    const float *fp16NeonValues = fp16NeonOut.DataAsFloat32();
    const float *fp16ScalarValues = fp16ScalarOut.DataAsFloat32();
    bool fp16Matches = fp16NeonValues != nullptr && fp16ScalarValues != nullptr;
    for (std::size_t index = 0; fp16Matches && index < 10U; ++index) {
      const float diff = fp16NeonValues[index] - fp16ScalarValues[index];
      fp16Matches = std::abs(diff) <= 1e-3F;
    }
    ok &= Expect(fp16Matches,
                 "neon matmul should match scalar outputs for fp16 inputs");

    us4::Tensor bf16MatmulLhs({3, 4}, us4::DType::kBFloat16,
                              us4::DeviceType::kCpu);
    us4::Tensor bf16MatmulRhs({4, 6}, us4::DType::kBFloat16,
                              us4::DeviceType::kCpu);
    us4::Tensor bf16NeonOut({3, 6}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    us4::Tensor bf16ScalarLhs({3, 4}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor bf16ScalarRhs({4, 6}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor bf16ScalarOut({3, 6}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    const std::vector<float> bf16LhsValues = {0.125F, 0.5F,   -0.75F, 1.25F,
                                              -1.0F,  0.375F, 0.875F, -0.5F,
                                              0.625F, -0.25F, 1.5F,   0.75F};
    const std::vector<float> bf16RhsValues = {
        0.25F, -0.5F, 0.75F,  1.0F,    -0.25F, 0.5F,    -0.75F, 0.125F,
        0.5F,  -1.0F, 0.25F,  0.875F,  1.125F, -0.625F, 0.375F, 0.25F,
        -0.5F, 0.75F, 0.625F, -0.125F, 1.0F,   -0.75F,  0.5F,   -0.25F};
    FillHalfTensor(bf16MatmulLhs, bf16LhsValues, true);
    FillHalfTensor(bf16MatmulRhs, bf16RhsValues, true);
    ok &= Expect(FillHalfReferenceTensor(bf16ScalarLhs, bf16LhsValues, true),
                 "scalar matmul should receive bf16-rounded lhs reference");
    ok &= Expect(FillHalfReferenceTensor(bf16ScalarRhs, bf16RhsValues, true),
                 "scalar matmul should receive bf16-rounded rhs reference");
    ok &= Expect(
        us4::NeonMatmul(bf16MatmulLhs, bf16MatmulRhs, bf16NeonOut, nullptr),
        "neon matmul should execute bf16 inputs");
    ok &= Expect(
        us4::ScalarMatmul(bf16ScalarLhs, bf16ScalarRhs, bf16ScalarOut, nullptr),
        "scalar matmul should provide bf16 reference");
    const float *bf16NeonValues = bf16NeonOut.DataAsFloat32();
    const float *bf16ScalarValues = bf16ScalarOut.DataAsFloat32();
    bool bf16Matches = bf16NeonValues != nullptr && bf16ScalarValues != nullptr;
    for (std::size_t index = 0; bf16Matches && index < 18U; ++index) {
      const float diff = bf16NeonValues[index] - bf16ScalarValues[index];
      bf16Matches = std::abs(diff) <= 1e-2F;
    }
    ok &= Expect(bf16Matches,
                 "neon matmul should match scalar outputs for bf16 inputs");

    us4::HardwareProbeResult narrowNeonProbe = neonProbe;
    narrowNeonProbe.neonVectorBits = 64;
    const us4::QwenAdapter qwenAdapter;
    const us4::BackendSelection narrowSelection = us4::SelectBackend(
        narrowNeonProbe, us4::RuntimeMode::kMicroPlus, qwenAdapter);
    ok &= Expect(narrowSelection.selected == us4::BackendType::kScalarCpu,
                 "narrow neon vectors should fall back to scalar");

    us4::Tensor matmulLhs({2, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor matmulRhs({3, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor matmulOut({2, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    float *lhsData = matmulLhs.MutableDataAsFloat32();
    float *rhsData = matmulRhs.MutableDataAsFloat32();
    lhsData[0] = 1.0F;
    lhsData[1] = 2.0F;
    lhsData[2] = 3.0F;
    lhsData[3] = 4.0F;
    lhsData[4] = 5.0F;
    lhsData[5] = 6.0F;
    rhsData[0] = 1.0F;
    rhsData[1] = 0.0F;
    rhsData[2] = 2.0F;
    rhsData[3] = 1.0F;
    rhsData[4] = 0.0F;
    rhsData[5] = 1.0F;
    rhsData[6] = 3.0F;
    rhsData[7] = 0.0F;
    rhsData[8] = 1.0F;
    rhsData[9] = 1.0F;
    rhsData[10] = 0.0F;
    rhsData[11] = 2.0F;
    ok &= Expect(us4::NeonMatmul(matmulLhs, matmulRhs, matmulOut, nullptr),
                 "neon matmul should execute fp32 fast path");
    const float *matmulValues = matmulOut.DataAsFloat32();
    ok &= Expect(matmulValues != nullptr && matmulValues[6] == 23.0F,
                 "neon matmul should preserve expected fp32 result");

    us4::Tensor tailLhs({3, 7}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor tailRhs({7, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor tailNeon({3, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor tailScalar({3, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    float *tailLhsData = tailLhs.MutableDataAsFloat32();
    float *tailRhsData = tailRhs.MutableDataAsFloat32();
    for (std::size_t index = 0; index < 21U; ++index) {
      tailLhsData[index] = static_cast<float>((index % 5U) - 2U);
    }
    for (std::size_t index = 0; index < 42U; ++index) {
      tailRhsData[index] = static_cast<float>((index % 7U) - 3U) * 0.5F;
    }
    ok &= Expect(us4::NeonMatmul(tailLhs, tailRhs, tailNeon, nullptr),
                 "neon matmul should handle tail columns");
    ok &= Expect(us4::ScalarMatmul(tailLhs, tailRhs, tailScalar, nullptr),
                 "scalar matmul should provide tail-column reference");
    const float *tailNeonValues = tailNeon.DataAsFloat32();
    const float *tailScalarValues = tailScalar.DataAsFloat32();
    bool tailMatches = tailNeonValues != nullptr && tailScalarValues != nullptr;
    for (std::size_t index = 0; tailMatches && index < 18U; ++index) {
      tailMatches = tailNeonValues[index] == tailScalarValues[index];
    }
    ok &= Expect(tailMatches,
                 "neon matmul should match scalar results for tail columns");

    us4::Tensor int8Weights({4}, us4::DType::kInt8, us4::DeviceType::kCpu);
    us4::Tensor int8Output({4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    auto *int8Bytes =
        reinterpret_cast<std::int8_t *>(int8Weights.MutableData());
    int8Bytes[0] = 4;
    int8Bytes[1] = -2;
    int8Bytes[2] = 3;
    int8Bytes[3] = -1;
    std::string error;
    ok &= Expect(us4::DequantizeInt8Groups(int8Weights, 2, {0.5F, 0.25F},
                                           int8Output, &error),
                 "neon int8 dequant should succeed for 2 groups");
    const float *int8Values = int8Output.DataAsFloat32();
    ok &= Expect(int8Values != nullptr && int8Values[2] == 0.75F,
                 "neon int8 dequant should scale the second group");

    us4::Tensor int4Weights({8}, us4::DType::kInt4, us4::DeviceType::kCpu);
    us4::Tensor int4Output({8}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    auto *int4Bytes =
        reinterpret_cast<std::uint8_t *>(int4Weights.MutableData());
    int4Bytes[0] = 0x2F;
    int4Bytes[1] = 0x91;
    int4Bytes[2] = 0x47;
    int4Bytes[3] = 0x8C;
    ok &= Expect(us4::DequantizeInt4Groups(int4Weights, 8, 4, {0.5F, 0.25F},
                                           int4Output, &error),
                 "neon int4 dequant should unpack signed nibbles");
    const float *int4Values = int4Output.DataAsFloat32();
    ok &= Expect(int4Values != nullptr && int4Values[7] == -2.0F,
                 "neon int4 dequant should preserve tail values");
  }

  {
    const us4::ContinuousBatcher singleBatcher(6U);
    const us4::BatchDecision singleDecision =
        singleBatcher.Schedule({{"solo", 9U, 1U, 0U}});
    ok &=
        Expect(singleDecision.singleSessionPassthrough,
               "continuous batcher should preserve single-session passthrough");
    ok &= Expect(
        singleDecision.totalGrantedTokens == 6U &&
            singleDecision.activeSessions == 1U &&
            singleDecision.slices.size() == 1U,
        "continuous batcher should cap single-session grant by batch size");
    ok &=
        Expect(singleDecision.slices.front().sessionId == "solo" &&
                   singleDecision.slices.front().grantedTokens == 6U &&
                   singleDecision.slices.front().roundsVisited == 1U,
               "continuous batcher should keep passthrough attribution stable");

    const us4::ContinuousBatcher fairBatcher(4U);
    const us4::BatchDecision fairDecision =
        fairBatcher.Schedule({{"alpha", 3U, 1U, 0U}, {"beta", 3U, 1U, 1U}});
    ok &= Expect(!fairDecision.singleSessionPassthrough,
                 "continuous batcher should disable passthrough for "
                 "multi-session scheduling");
    ok &= Expect(
        fairDecision.totalGrantedTokens == 4U &&
            fairDecision.activeSessions == 2U &&
            fairDecision.fairnessRounds == 2U,
        "continuous batcher should distribute full batch budget in two rounds");
    ok &= Expect(fairDecision.slices.size() == 2U &&
                     fairDecision.slices[0].sessionId == "alpha" &&
                     fairDecision.slices[0].grantedTokens == 2U &&
                     fairDecision.slices[1].sessionId == "beta" &&
                     fairDecision.slices[1].grantedTokens == 2U,
                 "continuous batcher should alternate equal sessions fairly");

    const us4::ContinuousBatcher weightedBatcher(6U);
    const us4::BatchDecision weightedDecision = weightedBatcher.Schedule(
        {{"heavy", 8U, 2U, 0U}, {"light", 8U, 1U, 1U}});
    ok &= Expect(weightedDecision.slices.size() == 2U &&
                     weightedDecision.slices[0].sessionId == "heavy" &&
                     weightedDecision.slices[0].grantedTokens == 4U &&
                     weightedDecision.slices[1].sessionId == "light" &&
                     weightedDecision.slices[1].grantedTokens == 2U,
                 "continuous batcher should honor fairness weight without "
                 "starving peers");
    ok &=
        Expect(weightedDecision.slices[0].roundsVisited == 2U &&
                   weightedDecision.slices[1].roundsVisited == 2U,
               "continuous batcher should surface rounds visited per session");
  }

  {
    const us4::Eagle3Decoder decoder(3U, 4U);
    const us4::Eagle3DraftTree tree =
        decoder.BuildTree({{4, 5, 6, 7}, {4, 5, 42, 8}, {1, 2, 3, 4}});
    const us4::Eagle3VerificationResult result =
        decoder.Verify({4, 5, 42, 9}, tree);
    ok &= Expect(tree.branches.size() == 3U,
                 "eagle3 decoder should preserve configured branch breadth");
    ok &= Expect(
        result.chosenBranchIndex == 1U && result.acceptedDepth == 3U &&
            result.rejectedBranches == 2U,
        "eagle3 decoder should pick the branch with the longest shared prefix");
    ok &= Expect(
        result.matchesAuthoritativePath &&
            result.committedTokens == std::vector<int>({4, 5, 42, 9}),
        "eagle3 decoder should remain equivalent to the authoritative path");
  }

  {
    const us4::PEagleDecoder decoder(4U);
    const us4::PEagleDraft acceptedDraft = decoder.Draft({7, 8, 9, 10, 11});
    const us4::PEagleVerificationResult acceptedResult =
        decoder.Verify({7, 8, 9, 10, 12}, acceptedDraft);
    ok &= Expect(acceptedDraft.tokens.size() == 4U,
                 "peagle decoder should clamp draft breadth");
    ok &= Expect(acceptedResult.acceptedCount == 4U &&
                     acceptedResult.rejectedCount == 0U &&
                     acceptedResult.allAccepted,
                 "peagle decoder should keep fully accepted drafts explicit");
    ok &= Expect(acceptedResult.matchesAuthoritativePath,
                 "peagle decoder should preserve authoritative equivalence "
                 "when accepted");

    const us4::PEagleDraft mismatchDraft = decoder.Draft({4, 5, 6, 7});
    const us4::PEagleVerificationResult mismatchResult =
        decoder.Verify({4, 5, 42, 8}, mismatchDraft);
    ok &= Expect(
        mismatchResult.acceptedCount == 2U &&
            mismatchResult.rejectedCount == 2U &&
            mismatchResult.fallbackToken.has_value() &&
            *mismatchResult.fallbackToken == 42,
        "peagle decoder should commit accepted prefix plus fallback token");
    ok &= Expect(
        mismatchResult.matchesAuthoritativePath &&
            mismatchResult.committedTokens == std::vector<int>({4, 5, 42}),
        "peagle decoder should stay equivalent to the authoritative path");
  }

  {
    us4::SessionPool pool;
    const std::string alphaKv = pool.NamespacedKvKey("alpha", "prompt");
    const std::string betaKv = pool.NamespacedKvKey("beta", "prompt");
    const std::string alphaPrefix =
        pool.NamespacedPrefixKey("alpha", "hello world");
    const std::string betaPrefix =
        pool.NamespacedPrefixKey("beta", "hello world");
    pool.RecordPromptPrefix("alpha", "hello world");
    pool.RecordPromptPrefix("beta", "hello world");
    const auto alpha = pool.Lookup("alpha");
    const auto beta = pool.Lookup("beta");

    ok &= Expect(alpha.has_value() && beta.has_value(),
                 "session pool should retain acquired sessions");
    ok &= Expect(alphaKv == "session::alpha::kv::prompt" &&
                     betaKv == "session::beta::kv::prompt",
                 "session pool should namespace kv keys per session");
    ok &= Expect(alphaPrefix != betaPrefix,
                 "session pool should isolate prompt prefixes across sessions");
    ok &= Expect(alpha->lastPromptPrefix == "hello world" &&
                     beta->lastPromptPrefix == "hello world",
                 "session pool should retain prompt ownership per session");
    ok &= Expect(pool.ActiveSessionCount() == 2U,
                 "session pool should report active session count");
    ok &= Expect(pool.Release("alpha") && !pool.Lookup("alpha").has_value() &&
                     pool.Lookup("beta").has_value(),
                 "session pool release should not leak other sessions");
  }

  {
    us4::Router router;
    const us4::RouterDecision prediction =
        router.RouteTopK({1.2F, 1.0F, 0.8F, 0.4F}, 3);
    const us4::RouterDecision actual =
        router.RouteTopK({1.1F, 0.2F, 0.9F, 0.8F}, 2);
    const us4::SpeculativePrefetch prefetch(3);
    const us4::SpeculativePrefetchPlan plan =
        prefetch.BuildPlan("glm", prediction);
    const us4::SpeculativePrefetchTelemetry telemetry =
        prefetch.Reconcile(plan, actual);

    ok &= Expect(plan.prefetchedExperts.size() == 3U,
                 "speculative prefetch should keep top-3 prediction breadth");
    ok &= Expect(plan.prefetchedKeys.front() == "glm-expert-0",
                 "speculative prefetch should keep family-scoped keys");
    ok &= Expect(telemetry.prefetchedCount == 3U && telemetry.hitCount == 2U &&
                     telemetry.missCount == 1U,
                 "speculative prefetch should expose hit and miss counts");
    ok &= Expect(std::abs(telemetry.hitRatio - (2.0 / 3.0)) <= 1e-9,
                 "speculative prefetch should expose stable hit ratio");
    ok &= Expect(telemetry.wrongExpertLeakPrevented,
                 "speculative prefetch should forbid wrong-expert leakage");
    ok &= Expect(
        telemetry.executableExperts.size() == 2U &&
            telemetry.executableExperts[0] == actual.selected[0].expert &&
            telemetry.executableExperts[1] == actual.selected[1].expert,
        "speculative prefetch should execute only actual route experts");
  }

  {
    us4::Router router;
    us4::SparsityAwareCache cache;
    const us4::RouterDecision routing =
        router.RouteTopK({1.4F, 1.1F, 0.3F, 0.2F}, 2);
    const us4::SparsityCacheSnapshot first = cache.Touch("glm", routing);
    const us4::SparsityCacheSnapshot second = cache.Touch("glm", routing);
    const us4::SparsityCacheSnapshot third = cache.Touch("minimax", routing);

    ok &= Expect(!first.lastLookupHit,
                 "first sparsity cache lookup should miss for a fresh route");
    ok &= Expect(
        second.lastLookupHit,
        "second sparsity cache lookup should hit for the same family pattern");
    ok &= Expect(second.hitCount == 1U,
                 "sparsity cache should count one hit after a repeated route");
    ok &= Expect(second.missCount == 1U,
                 "sparsity cache should preserve the first miss");
    ok &= Expect(
        std::abs(second.hitRatio - 0.5) <= 1e-9,
        "sparsity cache hit ratio should reflect hit over total lookups");
    ok &= Expect(third.entryCount == 2U,
                 "family-scoped sparsity cache entries should not collide");
    ok &= Expect(
        second.lastKey != third.lastKey,
        "different families should produce different sparsity cache keys");
  }

  {
    us4::MultimodalCache cache;
    const us4::MultimodalCacheSnapshot first = cache.Touch(
        "minimax",
        {{.modality = "text", .tokens = {"image", "audio", "fusion"}},
         {.modality = "image", .tokens = {"image"}},
         {.modality = "audio", .tokens = {"audio"}}});
    const us4::MultimodalCacheSnapshot second = cache.Touch(
        "minimax",
        {{.modality = "text", .tokens = {"image", "audio", "fusion"}},
         {.modality = "image", .tokens = {"image"}},
         {.modality = "audio", .tokens = {"audio"}}});
    const us4::MultimodalCacheSnapshot third = cache.Touch(
        "minimax",
        {{.modality = "text", .tokens = {"logic", "wide", "context"}},
         {.modality = "audio", .tokens = {"audio"}}});

    ok &= Expect(first.entryCount == 3U,
                 "multimodal cache should create one entry per modality state");
    ok &= Expect(first.lastTouchMisses == 3U && first.lastTouchHits == 0U,
                 "first multimodal cache touch should miss for each modality");
    ok &= Expect(
        second.lastTouchHits == 3U,
        "second multimodal cache touch should hit every repeated modality");
    ok &= Expect(
        std::abs(second.hitRatio - 0.5) <= 1e-9,
        "multimodal cache hit ratio should reflect repeated modality reuse");
    ok &= Expect(third.entryCount == 4U,
                 "new text state should add a new multimodal cache entry");
  }

  {
    const us4::IUS4V6Adapter *adapter =
        us4::FindAdapterByModel("deepseek-v2-lite");
    us4::RuntimeContext context(MakeProbe());
    adapter->ConfigureRuntime(context);
    const us4::GenerationResult result = adapter->Generate(
        {.prompt = "code logic runtime", .maxTokens = 3}, context);

    ok &= Expect(result.moePrefetchPrefetched == 3U,
                 "moe generation should surface speculative prefetch breadth");
    ok &= Expect(result.moePrefetchHits == 2U && result.moePrefetchMisses == 1U,
                 "moe generation should surface speculative prefetch hit and "
                 "miss counts");
    ok &= Expect(
        std::abs(result.moePrefetchHitRatio - (2.0 / 3.0)) <= 1e-9,
        "moe generation should surface stable speculative prefetch hit ratio");
    ok &=
        Expect(result.moePrefetchWrongExpertLeakPrevented,
               "moe generation should confirm wrong-expert leakage protection");
    ok &= Expect(
        result.moeSparsityCacheHitRatio == 0.0,
        "first moe generation should expose zero sparsity cache hit ratio");
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
