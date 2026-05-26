#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "core/rope.h"
#include "core/tensor.h"

namespace {

void Fill(us4::Tensor &tensor, const std::vector<float> &values) {
  ASSERT_EQ(tensor.ElementCount(), values.size());
  float *data = tensor.MutableDataAsFloat32();
  ASSERT_NE(data, nullptr);
  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = values[index];
  }
}

} // namespace

// Linear scaling is position interpolation: the effective rotation angle for
// pair 0 of a 2-wide tensor is position / scale (the lowest frequency band has
// inverse frequency theta^0 == 1, so only the scale divisor remains).
TEST(RopeContractTest, LinearScalingDividesPositionByScale) {
  constexpr float kTheta = 10000.0F;
  constexpr std::array<float, 3> kScales = {1.0F, 1.75F, 4.0F};

  for (const float scale : kScales) {
    SCOPED_TRACE(scale);
    us4::Tensor tensor({1, 2}, us4::DType::kFloat32);
    Fill(tensor, {1.0F, 0.0F});

    us4::ApplyRopeInPlace(tensor, 128U, kTheta, us4::RopeScalingType::kLinear,
                          scale);

    const float angle = 128.0F / scale;
    const float *data = tensor.DataAsFloat32();
    ASSERT_NE(data, nullptr);
    EXPECT_NEAR(data[0], std::cos(angle), 1e-4F);
    EXPECT_NEAR(data[1], std::sin(angle), 1e-4F);
  }
}

// Position 0 with unit scale is a no-op: every rotation angle is 0.
TEST(RopeContractTest, ZeroPositionUnitScaleIsIdentity) {
  const std::vector<float> input = {1.0F, -0.5F, 0.25F, 0.75F};
  us4::Tensor tensor({1, 4}, us4::DType::kFloat32);
  Fill(tensor, input);

  us4::ApplyRopeInPlace(tensor, 0U, 10000.0F, us4::RopeScalingType::kLinear,
                        1.0F);

  const float *data = tensor.DataAsFloat32();
  ASSERT_NE(data, nullptr);
  for (std::size_t index = 0; index < input.size(); ++index) {
    EXPECT_NEAR(data[index], input[index], 1e-6F) << index;
  }
}

// RoPE rotates each (x0, x1) pair, so it must preserve the per-pair L2 norm for
// every scaling mode and every row/position.
TEST(RopeContractTest, PreservesPerPairNormAcrossScalingModes) {
  constexpr std::size_t kRows = 2U;
  constexpr std::size_t kCols = 4U;
  const std::vector<float> input = {
      1.0F,   0.0F,   0.5F,   -0.25F,
      -0.75F, 0.125F, 0.875F, -0.5F,
  };
  constexpr std::array<us4::RopeScalingType, 3> kModes = {
      us4::RopeScalingType::kLinear,
      us4::RopeScalingType::kDynamic,
      us4::RopeScalingType::kYaRN,
  };

  for (const us4::RopeScalingType mode : kModes) {
    SCOPED_TRACE(static_cast<int>(mode));
    us4::Tensor tensor({kRows, kCols}, us4::DType::kFloat32);
    Fill(tensor, input);

    us4::ApplyRopeInPlace(tensor, 32U, 10000.0F, mode, 1.25F);

    const float *data = tensor.DataAsFloat32();
    ASSERT_NE(data, nullptr);
    for (std::size_t row = 0; row < kRows; ++row) {
      for (std::size_t pair = 0; pair < kCols / 2U; ++pair) {
        const std::size_t base = row * kCols + pair * 2U;
        const float inputNorm = std::hypot(input[base], input[base + 1]);
        const float rotatedNorm = std::hypot(data[base], data[base + 1]);
        EXPECT_NEAR(rotatedNorm, inputNorm, 1e-4F) << row << ":" << pair;
      }
    }
  }
}

// NTK-aware dynamic scaling rescales theta, so for scale > 1 it must diverge
// from plain linear scaling on the higher-frequency bands.
TEST(RopeContractTest, DynamicScalingDiffersFromLinearForScaleAboveOne) {
  constexpr std::size_t kCols = 8U;
  const std::vector<float> input(kCols, 1.0F);

  us4::Tensor linear({1, kCols}, us4::DType::kFloat32);
  us4::Tensor dynamic({1, kCols}, us4::DType::kFloat32);
  Fill(linear, input);
  Fill(dynamic, input);

  us4::ApplyRopeInPlace(linear, 64U, 10000.0F, us4::RopeScalingType::kLinear,
                        2.0F);
  us4::ApplyRopeInPlace(dynamic, 64U, 10000.0F, us4::RopeScalingType::kDynamic,
                        2.0F);

  const float *linearData = linear.DataAsFloat32();
  const float *dynamicData = dynamic.DataAsFloat32();
  ASSERT_NE(linearData, nullptr);
  ASSERT_NE(dynamicData, nullptr);

  float maxDelta = 0.0F;
  for (std::size_t index = 0; index < kCols; ++index) {
    maxDelta = std::max(maxDelta,
                        std::fabs(linearData[index] - dynamicData[index]));
  }
  EXPECT_GT(maxDelta, 1e-3F);
}
