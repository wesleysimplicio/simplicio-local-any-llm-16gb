#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace us4 {

// BitNet 1.58-bit packed weight layout. Each packed byte stores 5 ternary
// values (base-3) representing {-1, 0, +1}. The runtime decodes the byte at
// dispatch time and applies the per-row scale to the dot product.
struct BitNetPackedMatrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<std::uint8_t> packed;
  std::vector<float> rowScale;
};

// Decode a single packed byte into 5 ternit values. Used both by the NEON
// kernel scaffold and by the loader contract tests.
void DecodeBitNetByte(std::uint8_t packed, std::int8_t ternits[5]);

// Encode 5 ternit values ({-1, 0, +1}) into a single packed byte. Returns
// false when any value is outside the supported range.
bool EncodeBitNetByte(const std::int8_t ternits[5], std::uint8_t &packed);

// Reference NEON BitNet matmul. The implementation prefers the packed path
// but stays honest when the host has no NEON support, falling back to a
// scalar accumulation that still applies the row scale and ternary decoding.
//
// Returns the M x N output matrix in row-major fp32. The function leaves the
// `BitNetPackedMatrix` untouched and never throws.
std::vector<float> BitNetMatmul(const std::vector<float> &activations,
                                std::size_t activationRows,
                                const BitNetPackedMatrix &weights);

} // namespace us4
