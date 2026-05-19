#include "cpu/ternary_lut.h"

namespace us4 {

TernaryChunkValues DecodeTernaryChunk(const std::uint8_t packed) {
  TernaryChunkValues result;
  if (packed >= kTernaryLutSize) {
    return result;
  }
  std::uint8_t cursor = packed;
  for (std::size_t slot = 0; slot < kTernaryChunkSize; ++slot) {
    const std::uint8_t digit = static_cast<std::uint8_t>(cursor % 3U);
    cursor = static_cast<std::uint8_t>(cursor / 3U);
    result.values[slot] = static_cast<std::int8_t>(static_cast<int>(digit) - 1);
  }
  return result;
}

bool EncodeTernaryChunk(const TernaryChunkValues &values, std::uint8_t &packed) {
  std::uint8_t result = 0;
  std::uint8_t weight = 1;
  for (std::size_t slot = 0; slot < kTernaryChunkSize; ++slot) {
    const std::int8_t value = values.values[slot];
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

std::vector<TernaryChunkValues> BuildTernaryLookupTable() {
  std::vector<TernaryChunkValues> table(kTernaryLutSize);
  for (std::size_t packed = 0; packed < kTernaryLutSize; ++packed) {
    table[packed] = DecodeTernaryChunk(static_cast<std::uint8_t>(packed));
  }
  return table;
}

float TernaryChunkDot(const std::vector<TernaryChunkValues> &lut,
                      const std::uint8_t packed, const float activations[4]) {
  if (packed >= lut.size()) {
    return 0.0F;
  }
  const auto &chunk = lut[packed];
  float accumulator = 0.0F;
  for (std::size_t slot = 0; slot < kTernaryChunkSize; ++slot) {
    accumulator += activations[slot] * static_cast<float>(chunk.values[slot]);
  }
  return accumulator;
}

} // namespace us4
