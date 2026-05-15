#pragma once

#include <cstddef>

#include "core/tensor.h"

namespace us4 {

enum class RopeScalingType {
  kLinear,
  kDynamic,
  kYaRN,
};

void ApplyRopeInPlace(Tensor& tensor,
                      std::size_t position,
                      float theta,
                      RopeScalingType scaling = RopeScalingType::kLinear,
                      float scale = 1.0F);

}  // namespace us4
