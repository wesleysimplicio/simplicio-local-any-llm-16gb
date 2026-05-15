#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace us4 {

struct UnifiedAllocation {
  std::vector<std::byte> bytes;
  bool cpuVisible = true;
  bool gpuVisible = false;
};

class UnifiedAllocator {
 public:
  std::shared_ptr<UnifiedAllocation> Allocate(std::size_t byteCount, bool gpuVisible);
  std::size_t AllocationCount() const;
  std::size_t ResidentBytes() const;

 private:
  std::vector<std::shared_ptr<UnifiedAllocation>> allocations_;
};

}  // namespace us4
