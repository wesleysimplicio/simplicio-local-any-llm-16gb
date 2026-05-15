#include "memory/unified_allocator.h"

namespace us4 {

std::shared_ptr<UnifiedAllocation> UnifiedAllocator::Allocate(const std::size_t byteCount, const bool gpuVisible) {
  auto allocation = std::make_shared<UnifiedAllocation>();
  allocation->bytes.resize(byteCount);
  allocation->gpuVisible = gpuVisible;
  allocations_.push_back(allocation);
  return allocation;
}

std::size_t UnifiedAllocator::AllocationCount() const { return allocations_.size(); }

std::size_t UnifiedAllocator::ResidentBytes() const {
  std::size_t total = 0;
  for (const auto& allocation : allocations_) {
    total += allocation->bytes.size();
  }
  return total;
}

}  // namespace us4
