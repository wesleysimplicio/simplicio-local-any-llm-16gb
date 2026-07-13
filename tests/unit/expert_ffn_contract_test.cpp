#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "moe/expert_ffn.h"

namespace us4 {
namespace {

// Issue #81.7c: independent hand-computed oracle for the SwiGLU forward
// (down_proj(silu(gate_proj(x)) * up_proj(x))), not a reuse of the
// production math -- 2 hidden dims, 3 intermediate dims, chosen so every
// step's arithmetic is checkable by hand.
TEST(ExpertFfnContractTest, MatchesHandComputedSwigluOracle) {
  // x = [1, 2]
  const std::vector<float> x = {1.0F, 2.0F};

  ExpertFfnWeights weights;
  // gate_proj: [3, 2]
  weights.gate = {
      1.0F, 0.0F, // row 0: gate_out[0] = 1*x0 + 0*x1 = 1
      0.0F, 1.0F, // row 1: gate_out[1] = 0*x0 + 1*x1 = 2
      1.0F, 1.0F, // row 2: gate_out[2] = 1*x0 + 1*x1 = 3
  };
  weights.gateShape = {3, 2};
  // up_proj: [3, 2]
  weights.up = {
      2.0F, 0.0F, // up_out[0] = 2
      0.0F, 2.0F, // up_out[1] = 4
      1.0F, 0.0F, // up_out[2] = 1
  };
  weights.upShape = {3, 2};
  // down_proj: [2, 3]
  weights.down = {
      1.0F, 0.0F, 0.0F, // output[0] = hidden[0]
      0.0F, 0.0F, 1.0F, // output[1] = hidden[2]
  };
  weights.downShape = {2, 3};

  const auto silu = [](const double value) {
    return value / (1.0 + std::exp(-value));
  };
  const double hidden0 = silu(1.0) * 2.0;
  const double hidden2 = silu(3.0) * 1.0;
  const std::vector<double> expected = {hidden0, hidden2};

  const std::vector<float> actual = ApplyExpertFfnSwiglu(x, weights);
  ASSERT_EQ(actual.size(), 2U);
  EXPECT_NEAR(actual[0], expected[0], 1e-5);
  EXPECT_NEAR(actual[1], expected[1], 1e-5);
}

TEST(ExpertFfnContractTest, ZeroGateProducesZeroOutputRegardlessOfUp) {
  // silu(0) = 0, so hidden = 0 * up = 0 regardless of up_proj's values --
  // an all-zero gate_proj must always zero out that FFN's contribution.
  const std::vector<float> x = {1.0F, 1.0F};

  ExpertFfnWeights weights;
  weights.gate = {0.0F, 0.0F};
  weights.gateShape = {1, 2};
  weights.up = {5.0F, 5.0F};
  weights.upShape = {1, 2};
  weights.down = {3.0F, 4.0F};
  weights.downShape = {2, 1};

  const std::vector<float> actual = ApplyExpertFfnSwiglu(x, weights);
  ASSERT_EQ(actual.size(), 2U);
  EXPECT_FLOAT_EQ(actual[0], 0.0F);
  EXPECT_FLOAT_EQ(actual[1], 0.0F);
}

} // namespace
} // namespace us4
