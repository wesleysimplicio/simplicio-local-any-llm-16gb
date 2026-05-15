#pragma once

#include <cstddef>

#include "core/tensor.h"

namespace us4 {

enum class RopeScalingType {
  kLinear,
  kDynamic,
  kYaRN,
};

// Applies RoPE in-place over a contiguous [rows, cols] float32 tensor.
// Each row is treated as the next token position starting at `position`.
void ApplyRopeInPlace(Tensor &tensor, std::size_t position, float theta,
                      RopeScalingType scaling = RopeScalingType::kLinear,
                      float scale = 1.0F);

} // namespace us4
