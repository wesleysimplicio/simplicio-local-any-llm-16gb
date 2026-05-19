#include "ane/mixed_dispatch.h"

#include <array>
#include <utility>

namespace us4 {

std::string_view ToString(const DispatchBackend backend) {
  switch (backend) {
  case DispatchBackend::kMetal:
    return "metal";
  case DispatchBackend::kAne:
    return "ane";
  }
  return "metal";
}

MixedDispatchCoordinator::MixedDispatchCoordinator(
    const HardwareProbeResult &hardware)
    : available_(hardware.hasMetal && hardware.hasAne &&
                 hardware.supportsCoreMl) {
  if (!available_) {
    reason_ = "mixed-dispatch-unavailable";
    return;
  }
  reason_ = "mixed-dispatch-ready";
}

bool MixedDispatchCoordinator::Available() const { return available_; }

std::string_view MixedDispatchCoordinator::Reason() const { return reason_; }

MixedDispatchPlan MixedDispatchCoordinator::BuildPlan(
    std::string family, const std::size_t tokenCount,
    const std::size_t hiddenSize, const DType weightDType,
    const RuntimeMode mode) const {
  MixedDispatchPlan plan{.family = std::move(family)};
  const std::array<std::pair<std::string, OffloadLayerType>, 3> stages{{
      {"decoder.block.0.attention_qkv", OffloadLayerType::kAttentionQkv},
      {"decoder.block.0.attention_out", OffloadLayerType::kAttentionOutput},
      {"decoder.block.0.mlp_up", OffloadLayerType::kMlpUpProjection},
  }};

  const bool preferAne =
      available_ && mode == RuntimeMode::kFull && tokenCount > 0U &&
      tokenCount <= 64U && hiddenSize > 0U && hiddenSize % 8U == 0U &&
      weightDType != DType::kInt4 && weightDType != DType::kInt8;

  for (const auto &[layerName, layerType] : stages) {
    plan.stages.push_back(MixedDispatchStage{
        .layerName = layerName,
        .layerType = layerType,
        .backend = preferAne ? DispatchBackend::kAne : DispatchBackend::kMetal,
        .tokenCount = tokenCount,
        .hiddenSize = hiddenSize,
        .reason = preferAne ? "ane-layer-eligible" : "metal-fallback",
    });
  }

  return plan;
}

MixedDispatchTelemetry MixedDispatchCoordinator::Execute(
    const MixedDispatchPlan &plan, LayerOffloader &offloader,
    AneBackend &aneBackend, MetalCommandQueue &metalQueue,
    const std::shared_ptr<UnifiedAllocation> &allocation) const {
  MixedDispatchTelemetry telemetry;
  for (const MixedDispatchStage &stage : plan.stages) {
    if (stage.backend == DispatchBackend::kAne) {
      const OffloadDecision decision =
          offloader.Decide({.family = plan.family,
                            .layerName = stage.layerName,
                            .layerType = stage.layerType,
                            .mode = RuntimeMode::kFull,
                            .tokenCount = stage.tokenCount,
                            .hiddenSize = stage.hiddenSize,
                            .weightDType = DType::kFloat16,
                            .staticShape = true});
      if (decision.eligible &&
          aneBackend.Compile({.kind = AneModelKind::kAttentionMlp,
                              .family = plan.family,
                              .layerName = stage.layerName,
                              .tokenCount = stage.tokenCount,
                              .usesSharedTokenizer = false,
                              .staticShapePreferred = true}) &&
          aneBackend.Predict(stage.tokenCount, 1U)) {
        ++telemetry.aneStages;
        ++telemetry.compiledLayers;
        ++telemetry.predictionCalls;
        continue;
      }
    }

    if (metalQueue.Dispatch(MetalKernelKind::kMatmul,
                            stage.tokenCount > 0U ? stage.tokenCount : 1U, 32U,
                            allocation)) {
      ++telemetry.metalStages;
    }
  }

  if (telemetry.aneStages > 0U && telemetry.metalStages > 0U) {
    telemetry.strategy = "metal-ane-mixed";
  } else if (telemetry.aneStages > 0U) {
    telemetry.strategy = "ane-only";
  } else {
    telemetry.strategy = "metal-only";
  }
  return telemetry;
}

} // namespace us4
