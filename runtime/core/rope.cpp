#include "core/rope.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace us4 {

namespace {

constexpr double kDefaultTheta = 10000.0;

double SanitizeTheta(const float theta) {
  return theta > 1.0F ? static_cast<double>(theta) : kDefaultTheta;
}

double SanitizeScale(const float scale) {
  return scale > 1.0F ? static_cast<double>(scale) : 1.0;
}

double PairProgress(const std::size_t pairIndex, const std::size_t pairCount) {
  if (pairCount <= 1U) {
    return 0.0;
  }
  return static_cast<double>(pairIndex) /
         static_cast<double>(pairCount - 1U);
}

double ClampUnitRange(const double value) {
  return std::min(1.0, std::max(0.0, value));
}

double SmoothStep(const double value) {
  const double clamped = ClampUnitRange(value);
  return clamped * clamped * (3.0 - 2.0 * clamped);
}

double ComputeInverseFrequency(const double theta,
                               const std::size_t rotaryDim,
                               const std::size_t pairIndex) {
  const double exponent =
      -static_cast<double>(pairIndex * 2U) / static_cast<double>(rotaryDim);
  return std::exp(std::log(theta) * exponent);
}

double ComputeDynamicTheta(const double theta, const double scale,
                           const std::size_t rotaryDim) {
  if (scale <= 1.0 || rotaryDim <= 2U) {
    return theta;
  }

  const double exponent = static_cast<double>(rotaryDim) /
                          static_cast<double>(rotaryDim - 2U);
  return theta * std::pow(scale, exponent);
}

double ComputeYaRNInverseFrequency(const double theta, const double scale,
                                   const std::size_t rotaryDim,
                                   const std::size_t pairIndex,
                                   const std::size_t pairCount) {
  const double linearInverseFrequency =
      ComputeInverseFrequency(theta, rotaryDim, pairIndex) / scale;
  const double dynamicInverseFrequency = ComputeInverseFrequency(
      ComputeDynamicTheta(theta, scale, rotaryDim), rotaryDim, pairIndex);
  if (pairCount <= 1U) {
    return dynamicInverseFrequency;
  }

  // The current API only exposes theta + scale, so YaRN stays deterministic
  // here by blending the linear and dynamic frequency paths across dimensions.
  const double blend = SmoothStep(PairProgress(pairIndex, pairCount));
  return linearInverseFrequency +
         ((dynamicInverseFrequency - linearInverseFrequency) * blend);
}

std::vector<double> BuildInverseFrequencies(const RopeScalingType scaling,
                                            const double theta,
                                            const double scale,
                                            const std::size_t pairCount) {
  const std::size_t rotaryDim = pairCount * 2U;
  const double dynamicTheta = ComputeDynamicTheta(theta, scale, rotaryDim);

  std::vector<double> frequencies;
  frequencies.reserve(pairCount);
  for (std::size_t pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
    switch (scaling) {
    case RopeScalingType::kLinear:
      frequencies.push_back(
          ComputeInverseFrequency(theta, rotaryDim, pairIndex) / scale);
      break;
    case RopeScalingType::kDynamic:
      frequencies.push_back(
          ComputeInverseFrequency(dynamicTheta, rotaryDim, pairIndex));
      break;
    case RopeScalingType::kYaRN:
      frequencies.push_back(ComputeYaRNInverseFrequency(
          theta, scale, rotaryDim, pairIndex, pairCount));
      break;
    }
  }

  return frequencies;
}

} // namespace

void ApplyRopeInPlace(Tensor &tensor, const std::size_t position,
                      const float theta, const RopeScalingType scaling,
                      const float scale) {
  if (tensor.dtype() != DType::kFloat32 ||
      tensor.MutableDataAsFloat32() == nullptr || tensor.Rank() != 2 ||
      !tensor.IsContiguous()) {
    return;
  }

  float *data = tensor.MutableDataAsFloat32();
  const std::size_t rows = tensor.Shape()[0];
  const std::size_t cols = tensor.Shape()[1];
  const std::size_t pairCount = cols / 2U;
  if (pairCount == 0U) {
    return;
  }

  const double effectiveTheta = SanitizeTheta(theta);
  const double effectiveScale = SanitizeScale(scale);
  const std::vector<double> inverseFrequencies =
      BuildInverseFrequencies(scaling, effectiveTheta, effectiveScale,
                              pairCount);

  for (std::size_t row = 0; row < rows; ++row) {
    const double rowPosition = static_cast<double>(position + row);
    for (std::size_t pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
      const std::size_t col = pairIndex * 2U;
      const double angle = rowPosition * inverseFrequencies[pairIndex];
      const double cosine = std::cos(angle);
      const double sine = std::sin(angle);
      const std::size_t base = row * cols + col;
      const float x0 = data[base];
      const float x1 = data[base + 1];
      data[base] = static_cast<float>(static_cast<double>(x0) * cosine -
                                      static_cast<double>(x1) * sine);
      data[base + 1] = static_cast<float>(static_cast<double>(x0) * sine +
                                          static_cast<double>(x1) * cosine);
    }
  }
}

} // namespace us4
