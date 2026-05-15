#include "metal/command_queue.h"

#include "metal/autorelease_scope.h"
#include "metal/kernel_library.h"

namespace us4 {

std::string_view ToString(const MetalKernelKind kernel) {
  switch (kernel) {
  case MetalKernelKind::kMatmul:
    return "matmul";
  case MetalKernelKind::kSoftmax:
    return "softmax";
  case MetalKernelKind::kRmsNorm:
    return "rmsnorm";
  }
  return "matmul";
}

std::string_view ToString(const MetalInitializationStage stage) {
  switch (stage) {
  case MetalInitializationStage::kUnavailable:
    return "unavailable";
  case MetalInitializationStage::kDeviceReady:
    return "device-ready";
  case MetalInitializationStage::kQueueReady:
    return "queue-ready";
  }
  return "unavailable";
}

MetalCommandQueue::MetalCommandQueue(const HardwareProbeResult &hardware)
    : available_(hardware.hasMetal), device_(ProbeMetalDevice(hardware)) {
  profile_.requiresAutoreleaseBoundary =
      hardware.platform == "macos" || hardware.platform == "ios";
#if defined(__APPLE__)
  profile_.hostSupportsObjectiveCBoundary = true;
#endif

  if (!hardware.hasMetal || !device_.available) {
    available_ = false;
    reason_ = "metal-unavailable";
    profile_.stage = MetalInitializationStage::kUnavailable;
    return;
  }

  profile_.stage = MetalInitializationStage::kDeviceReady;
  reason_ = "metal-device-ready";

  if (!device_.queueLabel.empty() && device_.queueLabel != "disabled") {
    profile_.queueCreated = true;
    profile_.stage = MetalInitializationStage::kQueueReady;
    reason_ = "metal-queue-ready";
  }
  available_ = profile_.queueCreated;
}

bool MetalCommandQueue::Available() const { return available_; }

std::string_view MetalCommandQueue::Reason() const { return reason_; }

const MetalDeviceInfo &MetalCommandQueue::Device() const { return device_; }

const MetalQueueProfile &MetalCommandQueue::Profile() const { return profile_; }

bool MetalCommandQueue::Dispatch(
    const MetalKernelKind kernel, const std::size_t threadgroups,
    const std::size_t threadsPerGroup,
    const std::shared_ptr<UnifiedAllocation> &allocation) {
  if (!available_ || !profile_.queueCreated || threadgroups == 0 ||
      threadsPerGroup == 0) {
    return false;
  }

  const MetalKernelDescriptor *descriptor = FindMetalKernel(kernel);
  if (descriptor == nullptr) {
    reason_ = "metal-kernel-missing";
    return false;
  }

  const ScopedAutoreleasePool pool(profile_.requiresAutoreleaseBoundary);

  dispatches_.push_back(MetalDispatchRecord{
      .kernel = kernel,
      .entryPoint = descriptor->entryPoint,
      .relativePath = descriptor->relativePath,
      .threadgroups = threadgroups,
      .threadsPerGroup = threadsPerGroup,
      .usesSharedAllocation = allocation != nullptr && allocation->gpuVisible,
      .autoreleaseBoundaryRequested = pool.Requested(),
      .autoreleasePoolActive = pool.Active(),
      .autoreleaseBoundaryKind = ToString(pool.Kind()),
  });
  reason_ = "metal-dispatch-recorded";
  return true;
}

void MetalCommandQueue::Reset() {
  dispatches_.clear();
  if (profile_.stage == MetalInitializationStage::kQueueReady) {
    reason_ = "metal-queue-ready";
  } else if (profile_.stage == MetalInitializationStage::kDeviceReady) {
    reason_ = "metal-device-ready";
  }
}

std::size_t MetalCommandQueue::DispatchCount() const {
  return dispatches_.size();
}

const std::vector<MetalDispatchRecord> &MetalCommandQueue::Dispatches() const {
  return dispatches_;
}

} // namespace us4
