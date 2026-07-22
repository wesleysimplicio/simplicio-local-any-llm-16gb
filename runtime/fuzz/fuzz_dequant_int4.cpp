#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "neon/dequant_int4.h"

namespace {
std::int8_t DecodeNibble(const std::uint8_t nibble) {
  const std::uint8_t value = nibble & 0x0FU;
  return value >= 8U ? static_cast<std::int8_t>(value) - 16
                     : static_cast<std::int8_t>(value);
}
} // namespace

// The scalar oracle mirrors the packed int4 format but does not call any
// production dequantization helper. It covers odd tails and group boundaries.
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data,
                                      const std::size_t size) {
  if (data == nullptr || size == 0) {
    return 0;
  }

  const std::size_t logicalElementCount =
      std::min<std::size_t>(size, 2048U) * 2U;
  const std::size_t groupSize = 1U + (data[0] % 64U);
  us4::Tensor quantized({logicalElementCount}, us4::DType::kInt4);
  auto *encoded = reinterpret_cast<std::uint8_t *>(quantized.MutableData());
  for (std::size_t index = 0; index < quantized.ByteSize(); ++index) {
    encoded[index] = data[index % size];
  }
  std::vector<float> scales(
      (logicalElementCount + groupSize - 1U) / groupSize, 1.0F);
  for (std::size_t index = 0; index < logicalElementCount; ++index) {
    scales[index / groupSize] =
        0.125F + static_cast<float>(data[index % size] & 0x3FU) / 16.0F;
  }

  us4::Tensor output({logicalElementCount}, us4::DType::kFloat32);
  std::string error;
  if (!us4::DequantizeInt4Groups(quantized, logicalElementCount, groupSize,
                                 scales, output, &error)) {
    return 0;
  }
  const float *decoded = output.DataAsFloat32();
  for (std::size_t index = 0; index < logicalElementCount; ++index) {
    const std::uint8_t packed = encoded[index / 2U];
    const std::uint8_t nibble =
        index % 2U == 0U ? packed & 0x0FU : packed >> 4U;
    const float expected =
        static_cast<float>(DecodeNibble(nibble)) * scales[index / groupSize];
    if (!std::isfinite(decoded[index]) || decoded[index] != expected) {
      __builtin_trap();
    }
  }
  return 0;
}
