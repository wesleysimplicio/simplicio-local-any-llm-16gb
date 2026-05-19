#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace us4 {

// Ternary lookup table: 4-ternary chunks (each value in {-1, 0, +1}) are
// packed into a single byte using base-3 encoding. The LUT exposes a 256-entry
// table that pre-computes the dot product against four activations, so the
// runtime can dispatch ternary matmul with a single byte lookup instead of
// four conditional branches per chunk.

constexpr std::size_t kTernaryChunkSize = 4U;
constexpr std::size_t kTernaryLutSize = 81U; // 3^4 distinct chunks.

struct TernaryChunkValues {
  std::array<std::int8_t, kTernaryChunkSize> values = {0, 0, 0, 0};
};

// Decode a packed byte into 4 ternary values. Bytes above kTernaryLutSize - 1
// are treated as undefined and decoded to zeros.
TernaryChunkValues DecodeTernaryChunk(std::uint8_t packed);

// Encode 4 ternary values into a packed byte. Returns false when any value is
// outside the supported range.
bool EncodeTernaryChunk(const TernaryChunkValues &values, std::uint8_t &packed);

// Build the 81-entry lookup table. The result is exposed both as a flat
// std::vector for cache-friendly access and via a small accessor.
std::vector<TernaryChunkValues> BuildTernaryLookupTable();

// Compute the dot product between a packed ternary byte and four activations
// using the pre-built lookup table. Returns 0 for out-of-range packed values.
float TernaryChunkDot(const std::vector<TernaryChunkValues> &lut,
                      std::uint8_t packed, const float activations[4]);

} // namespace us4
