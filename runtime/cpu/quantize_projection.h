#pragma once

#include <cstddef>
#include <vector>

#include "core/tensor.h"

namespace us4 {

struct GroupwiseQuantizedProjection {
  Tensor tensor;
  std::vector<float> scales;
};

// Per-group scale = max(abs(source in group)) / maxQuantAbs, matching the
// int8 (maxQuantAbs=127) and int4 (maxQuantAbs=7) callers below.
std::vector<float> BuildGroupScales(const std::vector<float> &source,
                                    std::size_t groupSize, float maxQuantAbs);

GroupwiseQuantizedProjection
QuantizeProjectionInt8(const std::vector<float> &source,
                       const std::vector<std::size_t> &shape,
                       std::size_t groupSize);

GroupwiseQuantizedProjection
QuantizeProjectionInt4(const std::vector<float> &source,
                       const std::vector<std::size_t> &shape,
                       std::size_t groupSize);

} // namespace us4
