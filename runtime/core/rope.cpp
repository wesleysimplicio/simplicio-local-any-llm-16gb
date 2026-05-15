#include "core/rope.h"

#include <cmath>

namespace us4 {

namespace {

float EffectiveScale(const RopeScalingType scaling, const float scale, const std::size_t position) {
  switch (scaling) {
    case RopeScalingType::kLinear:
      return scale;
    case RopeScalingType::kDynamic:
      return scale * (1.0F + static_cast<float>(position) * 0.001F);
    case RopeScalingType::kYaRN:
      return scale * (1.0F + static_cast<float>(position) * 0.0005F);
  }
  return scale;
}

}  // namespace

void ApplyRopeInPlace(Tensor& tensor,
                      const std::size_t position,
                      const float theta,
                      const RopeScalingType scaling,
                      const float scale) {
  if (tensor.dtype() != DType::kFloat32 || tensor.MutableDataAsFloat32() == nullptr || tensor.Rank() != 2) {
    return;
  }

  float* data = tensor.MutableDataAsFloat32();
  const std::size_t rows = tensor.Shape()[0];
  const std::size_t cols = tensor.Shape()[1];
  const float effectiveScale = EffectiveScale(scaling, scale, position);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col + 1 < cols; col += 2) {
      const float frequency = std::pow(theta, -static_cast<float>(col) / static_cast<float>(cols));
      const float angle = static_cast<float>(position + row) * frequency * effectiveScale;
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      const std::size_t base = row * cols + col;
      const float x0 = data[base];
      const float x1 = data[base + 1];
      data[base] = x0 * cosine - x1 * sine;
      data[base + 1] = x0 * sine + x1 * cosine;
    }
  }
}

}  // namespace us4
