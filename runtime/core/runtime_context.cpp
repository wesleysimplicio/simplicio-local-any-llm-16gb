#include "core/runtime_context.h"

namespace us4 {

RuntimeContext::RuntimeContext(HardwareProbeResult probe_result)
    : hardware_(std::move(probe_result)), mode_(hardware_.recommendedMode),
      metalQueue_(hardware_), aneBackend_(hardware_),
      layerOffloader_(hardware_), mixedDispatch_(hardware_),
      mlxBridge_(hardware_), thermalMonitor_(hardware_) {
  mode_ = thermalMonitor_.Decide(mode_).effectiveMode;
}

const HardwareProbeResult &RuntimeContext::hardware() const {
  return hardware_;
}

RuntimeMode RuntimeContext::mode() const { return mode_; }

BackendType RuntimeContext::backend() const { return backend_; }

UnifiedAllocator &RuntimeContext::allocator() { return allocator_; }

const UnifiedAllocator &RuntimeContext::allocator() const { return allocator_; }

MetalCommandQueue &RuntimeContext::metalQueue() { return metalQueue_; }

const MetalCommandQueue &RuntimeContext::metalQueue() const {
  return metalQueue_;
}

AneBackend &RuntimeContext::aneBackend() { return aneBackend_; }

const AneBackend &RuntimeContext::aneBackend() const { return aneBackend_; }

LayerOffloader &RuntimeContext::layerOffloader() { return layerOffloader_; }

const LayerOffloader &RuntimeContext::layerOffloader() const {
  return layerOffloader_;
}

MixedDispatchCoordinator &RuntimeContext::mixedDispatch() {
  return mixedDispatch_;
}

const MixedDispatchCoordinator &RuntimeContext::mixedDispatch() const {
  return mixedDispatch_;
}

MlxBridge &RuntimeContext::mlxBridge() { return mlxBridge_; }

const MlxBridge &RuntimeContext::mlxBridge() const { return mlxBridge_; }

KvPager &RuntimeContext::kvPager() { return kvPager_; }

PrefixCache &RuntimeContext::prefixCache() { return prefixCache_; }

SsdColdStore &RuntimeContext::coldStore() { return coldStore_; }

Summarizer &RuntimeContext::summarizer() { return summarizer_; }

Router &RuntimeContext::router() { return router_; }

ExpertPager &RuntimeContext::expertPager() { return expertPager_; }

SparsityAwareCache &RuntimeContext::sparsityCache() { return sparsityCache_; }

MultimodalCache &RuntimeContext::multimodalCache() { return multimodalCache_; }

SessionPool &RuntimeContext::sessionPool() { return sessionPool_; }

ThermalMonitor &RuntimeContext::thermalMonitor() { return thermalMonitor_; }

const ThermalMonitor &RuntimeContext::thermalMonitor() const {
  return thermalMonitor_;
}

void RuntimeContext::SetMode(RuntimeMode mode) {
  mode_ = thermalMonitor_.Decide(mode).effectiveMode;
}

void RuntimeContext::SetBackend(BackendType backend) { backend_ = backend; }

} // namespace us4
