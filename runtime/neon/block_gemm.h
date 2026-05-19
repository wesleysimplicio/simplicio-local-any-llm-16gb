#pragma once

#include <cstddef>
#include <string_view>

#include "core/hardware_probe.h"
#include "core/tensor.h"

namespace us4 {

// BlockGemmTileShape describes the cache-aware tiling chosen by the runtime for
// a given matmul. The runtime currently uses the shape descriptively in
// telemetry and benchmark evidence; once toolchain-real NEON paths land the
// same descriptor drives the actual kernel dispatch.
struct BlockGemmTileShape {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t inner = 0;
  std::size_t prefetchDistance = 0;
  bool cacheAware = false;
  bool prefetchEnabled = false;
};

// SelectBlockGemmTileShape derives a deterministic cache-aware tile from the
// hardware probe and the operand layout. The returned shape encodes the intent
// of the tiled path even when execution is still routed through the existing
// scalar bridge. The decision must remain stable for identical inputs and
// honest about hosts where NEON is unavailable.
BlockGemmTileShape SelectBlockGemmTileShape(const HardwareProbeResult &hardware,
                                            const Tensor &lhs,
                                            const Tensor &rhs);

// FormatBlockGemmTileShape exposes a stable string representation so that the
// benchmark harness can include the tile descriptor in evidence files without
// duplicating formatting logic.
std::string_view FormatBlockGemmTileShape(const BlockGemmTileShape &shape,
                                          char *scratch, std::size_t capacity);

} // namespace us4
