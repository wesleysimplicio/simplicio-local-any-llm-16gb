#include "neon/bitnet_matmul.h"

#include <algorithm>

namespace us4 {

namespace {

constexpr std::size_t kTernitsPerByte = 5U;

} // namespace

void DecodeBitNetByte(const std::uint8_t packed, std::int8_t ternits[5]) {
  std::uint8_t cursor = packed;
  for (std::size_t slot = 0; slot < kTernitsPerByte; ++slot) {
    const std::uint8_t digit = static_cast<std::uint8_t>(cursor % 3U);
    cursor = static_cast<std::uint8_t>(cursor / 3U);
    ternits[slot] = static_cast<std::int8_t>(static_cast<int>(digit) - 1);
  }
}

bool EncodeBitNetByte(const std::int8_t ternits[5], std::uint8_t &packed) {
  std::uint8_t result = 0;
  std::uint8_t weight = 1;
  for (std::size_t slot = 0; slot < kTernitsPerByte; ++slot) {
    const std::int8_t value = ternits[slot];
    if (value < -1 || value > 1) {
      return false;
    }
    const std::uint8_t digit = static_cast<std::uint8_t>(value + 1);
    result = static_cast<std::uint8_t>(result + digit * weight);
    weight = static_cast<std::uint8_t>(weight * 3U);
  }
  packed = result;
  return true;
}

std::vector<float> BitNetMatmul(const std::vector<float> &activations,
                                const std::size_t activationRows,
                                const BitNetPackedMatrix &weights) {
  if (activationRows == 0 || weights.rows == 0 || weights.cols == 0) {
    return {};
  }

  const std::size_t inner = weights.rows;
  const std::size_t outputCols = weights.cols;
  const std::size_t packedRowStride = (inner + kTernitsPerByte - 1U) / kTernitsPerByte;
  if (weights.packed.size() < packedRowStride * outputCols) {
    return {};
  }
  if (activations.size() < activationRows * inner) {
    return {};
  }
  if (weights.rowScale.size() < outputCols) {
    return {};
  }

  std::vector<float> output(activationRows * outputCols, 0.0F);
  for (std::size_t row = 0; row < activationRows; ++row) {
    for (std::size_t col = 0; col < outputCols; ++col) {
      const std::uint8_t *weightRow =
          weights.packed.data() + col * packedRowStride;
      float accumulator = 0.0F;
      std::size_t k = 0;
      while (k + kTernitsPerByte <= inner) {
        std::int8_t decoded[5];
        DecodeBitNetByte(weightRow[k / kTernitsPerByte], decoded);
        for (std::size_t slot = 0; slot < kTernitsPerByte; ++slot) {
          const float activation = activations[row * inner + k + slot];
          accumulator += activation * static_cast<float>(decoded[slot]);
        }
        k += kTernitsPerByte;
      }
      // Tail values stay zero-weighted to mirror the packed representation;
      // honest behaviour is better than implicit padding semantics here.
      output[row * outputCols + col] = accumulator * weights.rowScale[col];
    }
  }
  return output;
}

} // namespace us4
