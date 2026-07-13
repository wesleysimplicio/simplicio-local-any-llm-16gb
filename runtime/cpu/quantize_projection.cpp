#include "cpu/quantize_projection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace us4 {

namespace {
std::uint8_t EncodeSignedNibble(const std::int8_t value) {
  return static_cast<std::uint8_t>(value < 0 ? value + 16 : value) & 0x0FU;
}
} // namespace

std::vector<float> BuildGroupScales(const std::vector<float> &source,
                                    const std::size_t groupSize,
                                    const float maxQuantAbs) {
  const std::size_t groupCount = (source.size() + groupSize - 1U) / groupSize;
  std::vector<float> scales(groupCount, 1.0F);
  for (std::size_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
    const std::size_t begin = groupIndex * groupSize;
    const std::size_t end = std::min(source.size(), begin + groupSize);
    float maxAbs = 0.0F;
    for (std::size_t index = begin; index < end; ++index) {
      maxAbs = std::max(maxAbs, std::fabs(source[index]));
    }
    scales[groupIndex] = maxAbs > std::numeric_limits<float>::epsilon()
                             ? maxAbs / maxQuantAbs
                             : 1.0F;
  }
  return scales;
}

GroupwiseQuantizedProjection
QuantizeProjectionInt8(const std::vector<float> &source,
                       const std::vector<std::size_t> &shape,
                       const std::size_t groupSize) {
  GroupwiseQuantizedProjection projection{
      .tensor = Tensor(shape, DType::kInt8),
      .scales = BuildGroupScales(source, groupSize, 127.0F),
  };

  auto *bytes =
      reinterpret_cast<std::int8_t *>(projection.tensor.MutableData());
  for (std::size_t index = 0; index < source.size(); ++index) {
    const std::size_t groupIndex = index / groupSize;
    const float scale = projection.scales[groupIndex];
    const float normalized = scale > std::numeric_limits<float>::epsilon()
                                 ? source[index] / scale
                                 : 0.0F;
    const long rounded = std::lround(normalized);
    bytes[index] =
        static_cast<std::int8_t>(std::clamp<long>(rounded, -127L, 127L));
  }

  return projection;
}

GroupwiseQuantizedProjection
QuantizeProjectionInt4(const std::vector<float> &source,
                       const std::vector<std::size_t> &shape,
                       const std::size_t groupSize) {
  GroupwiseQuantizedProjection projection{
      .tensor = Tensor(shape, DType::kInt4),
      .scales = BuildGroupScales(source, groupSize, 7.0F),
  };

  auto *bytes =
      reinterpret_cast<std::uint8_t *>(projection.tensor.MutableData());
  std::fill(bytes, bytes + projection.tensor.ByteSize(), 0U);
  for (std::size_t index = 0; index < source.size(); ++index) {
    const std::size_t groupIndex = index / groupSize;
    const float scale = projection.scales[groupIndex];
    const float normalized = scale > std::numeric_limits<float>::epsilon()
                                 ? source[index] / scale
                                 : 0.0F;
    const long rounded = std::lround(normalized);
    const std::int8_t clamped =
        static_cast<std::int8_t>(std::clamp<long>(rounded, -8L, 7L));
    const std::uint8_t nibble = EncodeSignedNibble(clamped);
    const std::size_t byteIndex = index / 2U;
    if (index % 2U == 0U) {
      bytes[byteIndex] =
          static_cast<std::uint8_t>((bytes[byteIndex] & 0xF0U) | nibble);
    } else {
      bytes[byteIndex] = static_cast<std::uint8_t>((bytes[byteIndex] & 0x0FU) |
                                                   (nibble << 4U));
    }
  }

  return projection;
}

} // namespace us4
