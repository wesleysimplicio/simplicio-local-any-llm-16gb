#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/hardware_probe.h"
#include "memory/unified_allocator.h"
#include "metal/device_info.h"

namespace us4 {

enum class MetalKernelKind {
  kMatmul,
  kSoftmax,
  kRmsNorm,
};

enum class MetalInitializationStage {
  kUnavailable,
  kDeviceReady,
  kQueueReady,
};

struct MetalDispatchRecord {
  MetalKernelKind kernel = MetalKernelKind::kMatmul;
  std::string_view entryPoint;
  std::string_view relativePath;
  std::size_t threadgroups = 0;
  std::size_t threadsPerGroup = 0;
  bool usesSharedAllocation = false;
  bool autoreleaseBoundaryRequested = false;
  bool autoreleasePoolActive = false;
  std::string_view autoreleaseBoundaryKind;
};

struct MetalQueueProfile {
  MetalInitializationStage stage = MetalInitializationStage::kUnavailable;
  bool queueCreated = false;
  bool requiresAutoreleaseBoundary = false;
  bool hostSupportsObjectiveCBoundary = false;
};

std::string_view ToString(MetalKernelKind kernel);
std::string_view ToString(MetalInitializationStage stage);

class MetalCommandQueue {
public:
  MetalCommandQueue() = default;
  explicit MetalCommandQueue(const HardwareProbeResult &hardware);

  bool Available() const;
  std::string_view Reason() const;
  const MetalDeviceInfo &Device() const;
  const MetalQueueProfile &Profile() const;
  bool Dispatch(MetalKernelKind kernel, std::size_t threadgroups,
                std::size_t threadsPerGroup,
                const std::shared_ptr<UnifiedAllocation> &allocation = nullptr);
  void Reset();
  std::size_t DispatchCount() const;
  const std::vector<MetalDispatchRecord> &Dispatches() const;

private:
  bool available_ = false;
  std::string reason_ = "metal-unavailable";
  MetalDeviceInfo device_;
  MetalQueueProfile profile_;
  std::vector<MetalDispatchRecord> dispatches_;
};

} // namespace us4
