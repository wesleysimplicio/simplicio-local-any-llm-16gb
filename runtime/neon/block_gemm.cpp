#include "neon/block_gemm.h"

#include <cstdio>

namespace us4 {

namespace {

constexpr std::size_t kDefaultPrefetchDistance = 64;

bool IsBlockEligible(const HardwareProbeResult &hardware) {
  return hardware.hasNeon && hardware.architecture == "arm64";
}

std::size_t LargestPowerOfTwoNotAbove(const std::size_t target,
                                      const std::size_t bound) {
  std::size_t candidate = bound;
  while (candidate > 1U && candidate > target) {
    candidate >>= 1U;
  }
  return candidate;
}

} // namespace

BlockGemmTileShape SelectBlockGemmTileShape(const HardwareProbeResult &hardware,
                                            const Tensor &lhs,
                                            const Tensor &rhs) {
  BlockGemmTileShape shape;
  if (!IsBlockEligible(hardware) || lhs.Empty() || rhs.Empty() ||
      lhs.Rank() != 2 || rhs.Rank() != 2) {
    return shape;
  }

  const std::size_t lhsRows = lhs.Shape()[0];
  const std::size_t lhsCols = lhs.Shape()[1];
  const std::size_t rhsCols = rhs.Shape()[1];

  // Default 8x8 tile aligns with the existing kernel profile flavors. The
  // alternative 4x16 shape is selected when the row dimension is short and the
  // output is wide, which matches the documented BLAS-style preferences in the
  // Sprint 04 plan.
  std::size_t tileRows = 8;
  std::size_t tileCols = 8;
  if (lhsRows < 4) {
    tileRows = 4;
    tileCols = 16;
  }

  shape.rows = LargestPowerOfTwoNotAbove(tileRows, lhsRows == 0 ? 1U : lhsRows);
  shape.cols = LargestPowerOfTwoNotAbove(tileCols, rhsCols == 0 ? 1U : rhsCols);
  shape.inner = LargestPowerOfTwoNotAbove(8U, lhsCols == 0 ? 1U : lhsCols);
  shape.prefetchDistance = kDefaultPrefetchDistance;
  shape.cacheAware = true;
  shape.prefetchEnabled = lhsCols >= 16U;
  return shape;
}

std::string_view FormatBlockGemmTileShape(const BlockGemmTileShape &shape,
                                          char *scratch,
                                          const std::size_t capacity) {
  if (scratch == nullptr || capacity == 0) {
    return {};
  }
  const int written = std::snprintf(
      scratch, capacity, "%zux%zux%zu prefetch=%zu cache=%s",
      shape.rows, shape.cols, shape.inner, shape.prefetchDistance,
      shape.cacheAware ? "on" : "off");
  if (written <= 0) {
    return {};
  }
  const auto safeWritten =
      static_cast<std::size_t>(written) < capacity
          ? static_cast<std::size_t>(written)
          : capacity - 1U;
  return std::string_view(scratch, safeWritten);
}

} // namespace us4
