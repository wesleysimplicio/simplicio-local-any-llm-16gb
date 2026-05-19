#pragma once

#include <cstddef>

namespace us4 {

// Speculative decoding telemetry. The struct is intentionally small so the
// scheduler can fill it on every batch step without overhead. All fields are
// CLI/bench-visible.

struct SpeculativeTelemetry {
  std::size_t draftAttempts = 0;
  std::size_t acceptedTokens = 0;
  std::size_t rejectedTokens = 0;
  std::size_t verifyWindow = 0;
  float acceptanceRate = 0.0F;
};

SpeculativeTelemetry ComputeSpeculativeTelemetry(std::size_t draftAttempts,
                                                 std::size_t acceptedTokens,
                                                 std::size_t verifyWindow);

}  // namespace us4
