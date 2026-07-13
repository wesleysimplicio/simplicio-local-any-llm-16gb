#include "moe/expert_ffn.h"

#include <cmath>

namespace us4 {

namespace {

float Silu(const float value) { return value / (1.0F + std::exp(-value)); }

} // namespace

std::vector<float> ApplyExpertFfnSwiglu(const std::vector<float> &x,
                                        const ExpertFfnWeights &weights) {
  const std::size_t hiddenSize = x.size();
  const std::size_t intermediateSize = weights.gateShape[0];

  std::vector<float> gateOut(intermediateSize, 0.0F);
  std::vector<float> upOut(intermediateSize, 0.0F);
  for (std::size_t row = 0; row < intermediateSize; ++row) {
    float gateSum = 0.0F;
    float upSum = 0.0F;
    for (std::size_t col = 0; col < hiddenSize; ++col) {
      const float input = x[col];
      gateSum += weights.gate[(row * hiddenSize) + col] * input;
      upSum += weights.up[(row * hiddenSize) + col] * input;
    }
    gateOut[row] = gateSum;
    upOut[row] = upSum;
  }

  std::vector<float> hidden(intermediateSize, 0.0F);
  for (std::size_t index = 0; index < intermediateSize; ++index) {
    hidden[index] = Silu(gateOut[index]) * upOut[index];
  }

  std::vector<float> output(hiddenSize, 0.0F);
  for (std::size_t row = 0; row < hiddenSize; ++row) {
    float sum = 0.0F;
    for (std::size_t col = 0; col < intermediateSize; ++col) {
      sum += weights.down[(row * intermediateSize) + col] * hidden[col];
    }
    output[row] = sum;
  }

  return output;
}

} // namespace us4
