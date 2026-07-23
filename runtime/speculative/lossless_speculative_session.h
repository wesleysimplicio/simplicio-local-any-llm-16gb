#pragma once

#include <cstddef>
#include <stop_token>
#include <vector>

#include "speculative/peagle_decoder.h"

namespace us4 {

enum class SpeculativeStopReason {
  kNone,
  kCancelled,
  kRoundLimit,
  kTokenLimit,
};

struct LosslessSpeculativeLimits {
  std::size_t maxDraftTokens = 4U;
  std::size_t maxRounds = 8U;
  std::size_t maxCommittedTokens = 32U;
};

struct LosslessSpeculativeMetrics {
  std::size_t rounds = 0U;
  std::size_t attemptedTokens = 0U;
  std::size_t acceptedTokens = 0U;
  std::size_t rejectedTokens = 0U;
  std::size_t committedTokens = 0U;
  std::size_t cancelledRounds = 0U;
  std::size_t limitedRounds = 0U;
  double acceptanceRate = 0.0;
};

struct LosslessSpeculativeRound {
  std::vector<int> committedTokens;
  SpeculativeStopReason stopReason = SpeculativeStopReason::kNone;
  bool matchesAuthoritativePath = true;
};

class LosslessSpeculativeSession {
public:
  explicit LosslessSpeculativeSession(LosslessSpeculativeLimits limits = {});

  [[nodiscard]] LosslessSpeculativeRound
  RunRound(const std::vector<int> &authoritativeTokens,
           const std::vector<int> &proposalTokens,
           std::stop_token stopToken = {});
  [[nodiscard]] const LosslessSpeculativeMetrics &Metrics() const noexcept;

private:
  LosslessSpeculativeLimits limits_;
  LosslessSpeculativeMetrics metrics_;
};

} // namespace us4
