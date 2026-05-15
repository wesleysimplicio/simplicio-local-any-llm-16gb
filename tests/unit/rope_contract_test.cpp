#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "core/rope.h"
#include "core/tensor.h"

namespace {

struct RopeScalingExpectation {
  const char *name;
  us4::RopeScalingType scaling;
  std::size_t position;
  float scale;
};

float ComputeEffectiveScale(const us4::RopeScalingType scaling,
                            const float scale,
                            const std::size_t position) {
  switch (scaling) {
  case us4::RopeScalingType::kLinear:
    return scale;
  case us4::RopeScalingType::kDynamic:
    return scale * (1.0F + static_cast<float>(position) * 0.001F);
  case us4::RopeScalingType::kYaRN:
    return scale * (1.0F + static_cast<float>(position) * 0.0005F);
  }

  return scale;
}

void Fill(us4::Tensor &tensor, const std::vector<float> &values) {
  ASSERT_EQ(tensor.ElementCount(), values.size());
  float *data = tensor.MutableDataAsFloat32();
  ASSERT_NE(data, nullptr);

  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = values[index];
  }
}

std::vector<float> ApplyReferenceRope(const std::vector<float> &input,
                                      const std::size_t rows,
                                      const std::size_t cols,
                                      const std::size_t position,
                                      const float theta,
                                      const us4::RopeScalingType scaling,
                                      const float scale) {
  std::vector<float> output = input;
  const float effectiveScale =
      ComputeEffectiveScale(scaling, scale, position);

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col + 1 < cols; col += 2) {
      const float frequency = std::pow(
          theta, -static_cast<float>(col) / static_cast<float>(cols));
      const float angle =
          static_cast<float>(position + row) * frequency * effectiveScale;
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      const std::size_t base = row * cols + col;
      const float x0 = output[base];
      const float x1 = output[base + 1];
      output[base] = x0 * cosine - x1 * sine;
      output[base + 1] = x0 * sine + x1 * cosine;
    }
  }

  return output;
}

} // namespace

TEST(RopeContractTest, AppliesExpectedScalingFactorForLinearDynamicAndYaRN) {
  constexpr float kTheta = 10000.0F;
  constexpr std::array<RopeScalingExpectation, 3> kCases = {{
      {"linear", us4::RopeScalingType::kLinear, 128U, 1.75F},
      {"dynamic", us4::RopeScalingType::kDynamic, 128U, 1.75F},
      {"yarn", us4::RopeScalingType::kYaRN, 128U, 1.75F},
  }};

  for (const RopeScalingExpectation &expectation : kCases) {
    SCOPED_TRACE(expectation.name);

    us4::Tensor tensor({1, 2}, us4::DType::kFloat32);
    Fill(tensor, {1.0F, 0.0F});

    us4::ApplyRopeInPlace(tensor, expectation.position, kTheta,
                          expectation.scaling, expectation.scale);

    const float angle = static_cast<float>(expectation.position) *
                        ComputeEffectiveScale(expectation.scaling,
                                              expectation.scale,
                                              expectation.position);
    const float *data = tensor.DataAsFloat32();
    ASSERT_NE(data, nullptr);
    EXPECT_NEAR(data[0], std::cos(angle), 1e-5F);
    EXPECT_NEAR(data[1], std::sin(angle), 1e-5F);
  }
}

TEST(RopeContractTest, MatchesReferenceTensorAcrossScalingModes) {
  constexpr float kTheta = 10000.0F;
  constexpr std::size_t kRows = 2U;
  constexpr std::size_t kCols = 4U;
  const std::vector<float> input = {
      1.0F,  0.0F,   0.5F,  -0.25F,
      -0.75F, 0.125F, 0.875F, -0.5F,
  };
  constexpr std::array<RopeScalingExpectation, 3> kCases = {{
      {"linear", us4::RopeScalingType::kLinear, 32U, 1.25F},
      {"dynamic", us4::RopeScalingType::kDynamic, 32U, 1.25F},
      {"yarn", us4::RopeScalingType::kYaRN, 32U, 1.25F},
  }};

  for (const RopeScalingExpectation &expectation : kCases) {
    SCOPED_TRACE(expectation.name);

    us4::Tensor tensor({kRows, kCols}, us4::DType::kFloat32);
    Fill(tensor, input);

    us4::ApplyRopeInPlace(tensor, expectation.position, kTheta,
                          expectation.scaling, expectation.scale);

    const std::vector<float> expected =
        ApplyReferenceRope(input, kRows, kCols, expectation.position, kTheta,
                           expectation.scaling, expectation.scale);
    const float *data = tensor.DataAsFloat32();
    ASSERT_NE(data, nullptr);

    for (std::size_t index = 0; index < expected.size(); ++index) {
      EXPECT_NEAR(data[index], expected[index], 1e-5F) << index;
    }
  }
}
