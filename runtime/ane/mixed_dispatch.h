#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ane/ane_backend.h"
#include "ane/layer_offloader.h"
#include "memory/unified_allocator.h"
#include "metal/command_queue.h"

namespace us4 {

enum class DispatchBackend {
  kMetal,
  kAne,
};

struct MixedDispatchStage {
  std::string layerName;
  OffloadLayerType layerType = OffloadLayerType::kEmbedding;
  DispatchBackend backend = DispatchBackend::kMetal;
  std::size_t tokenCount = 0;
  std::size_t hiddenSize = 0;
  std::string reason = "metal-default";
};

struct MixedDispatchPlan {
  std::string family;
  std::vector<MixedDispatchStage> stages;
};

struct MixedDispatchTelemetry {
  std::size_t metalStages = 0;
  std::size_t aneStages = 0;
  std::size_t compiledLayers = 0;
  std::size_t predictionCalls = 0;
  std::string strategy = "metal-only";
};

std::string_view ToString(DispatchBackend backend);

class MixedDispatchCoordinator {
public:
  MixedDispatchCoordinator() = default;
  explicit MixedDispatchCoordinator(const HardwareProbeResult &hardware);

  bool Available() const;
  std::string_view Reason() const;
  MixedDispatchPlan BuildPlan(std::string family, std::size_t tokenCount,
                              std::size_t hiddenSize, DType weightDType,
                              RuntimeMode mode) const;
  MixedDispatchTelemetry
  Execute(const MixedDispatchPlan &plan, LayerOffloader &offloader,
          AneBackend &aneBackend, MetalCommandQueue &metalQueue,
          const std::shared_ptr<UnifiedAllocation> &allocation) const;

private:
  bool available_ = false;
  std::string reason_ = "mixed-dispatch-unavailable";
};

} // namespace us4
