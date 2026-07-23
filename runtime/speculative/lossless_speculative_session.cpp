#include "speculative/lossless_speculative_session.h"

#include <algorithm>

namespace us4 {

LosslessSpeculativeSession::LosslessSpeculativeSession(
    LosslessSpeculativeLimits limits)
    : limits_(limits) {
  limits_.maxDraftTokens =
      std::max<std::size_t>(1U, limits_.maxDraftTokens);
}

LosslessSpeculativeRound LosslessSpeculativeSession::RunRound(
    const std::vector<int> &authoritativeTokens,
    const std::vector<int> &proposalTokens, const std::stop_token stopToken) {
  LosslessSpeculativeRound round;
  if (stopToken.stop_requested()) {
    round.stopReason = SpeculativeStopReason::kCancelled;
    ++metrics_.cancelledRounds;
    return round;
  }
  if (metrics_.rounds >= limits_.maxRounds) {
    round.stopReason = SpeculativeStopReason::kRoundLimit;
    ++metrics_.limitedRounds;
    return round;
  }
  if (metrics_.committedTokens >= limits_.maxCommittedTokens) {
    round.stopReason = SpeculativeStopReason::kTokenLimit;
    ++metrics_.limitedRounds;
    return round;
  }

  ++metrics_.rounds;
  const PEagleDecoder decoder(limits_.maxDraftTokens);
  const PEagleDraft draft = decoder.Draft(proposalTokens);
  const PEagleVerificationResult verified =
      decoder.Verify(authoritativeTokens, draft);
  metrics_.attemptedTokens += draft.tokens.size();
  metrics_.acceptedTokens += verified.acceptedCount;
  metrics_.rejectedTokens += verified.rejectedCount;

  const std::size_t remaining =
      limits_.maxCommittedTokens - metrics_.committedTokens;
  const std::size_t commitCount =
      std::min<std::size_t>(remaining, verified.committedTokens.size());
  round.committedTokens.assign(
      verified.committedTokens.begin(),
      verified.committedTokens.begin() +
          static_cast<std::ptrdiff_t>(commitCount));
  metrics_.committedTokens += commitCount;
  round.matchesAuthoritativePath =
      round.committedTokens.size() <= authoritativeTokens.size() &&
      std::equal(round.committedTokens.begin(), round.committedTokens.end(),
                 authoritativeTokens.begin());
  if (commitCount < verified.committedTokens.size() ||
      metrics_.committedTokens >= limits_.maxCommittedTokens) {
    round.stopReason = SpeculativeStopReason::kTokenLimit;
  }
  metrics_.acceptanceRate =
      metrics_.attemptedTokens == 0U
          ? 0.0
          : static_cast<double>(metrics_.acceptedTokens) /
                static_cast<double>(metrics_.attemptedTokens);
  return round;
}

const LosslessSpeculativeMetrics &
LosslessSpeculativeSession::Metrics() const noexcept {
  return metrics_;
}

} // namespace us4
