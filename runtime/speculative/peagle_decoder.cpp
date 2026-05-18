#include "speculative/peagle_decoder.h"

#include <algorithm>

namespace us4 {

PEagleDecoder::PEagleDecoder(const std::size_t maxDraftTokens)
    : maxDraftTokens_(std::max<std::size_t>(1U, maxDraftTokens)) {}

std::size_t PEagleDecoder::MaxDraftTokens() const noexcept {
  return maxDraftTokens_;
}

PEagleDraft PEagleDecoder::Draft(const std::vector<int> &proposalTokens) const {
  PEagleDraft draft;
  const std::size_t count =
      std::min<std::size_t>(maxDraftTokens_, proposalTokens.size());
  draft.tokens.assign(proposalTokens.begin(), proposalTokens.begin() + count);
  return draft;
}

PEagleVerificationResult
PEagleDecoder::Verify(const std::vector<int> &authoritativeTokens,
                      const PEagleDraft &draft) const {
  PEagleVerificationResult result;
  if (draft.tokens.empty()) {
    result.allAccepted = true;
    result.matchesAuthoritativePath = true;
    return result;
  }

  const std::size_t comparable =
      std::min(authoritativeTokens.size(), draft.tokens.size());
  std::size_t accepted = 0U;
  while (accepted < comparable &&
         authoritativeTokens[accepted] == draft.tokens[accepted]) {
    result.committedTokens.push_back(draft.tokens[accepted]);
    ++accepted;
  }

  result.acceptedCount = accepted;
  result.allAccepted = accepted == draft.tokens.size() &&
                       authoritativeTokens.size() >= draft.tokens.size();

  if (result.allAccepted) {
    result.rejectedCount = 0U;
  } else {
    if (accepted < authoritativeTokens.size()) {
      result.fallbackToken = authoritativeTokens[accepted];
      result.committedTokens.push_back(*result.fallbackToken);
    }
    result.rejectedCount = draft.tokens.size() - accepted;
  }

  if (!draft.tokens.empty()) {
    result.acceptanceRate = static_cast<double>(result.acceptedCount) /
                            static_cast<double>(draft.tokens.size());
  }

  if (accepted == 0U && authoritativeTokens.empty()) {
    result.matchesAuthoritativePath = result.committedTokens.empty();
  } else if (result.committedTokens.size() <= authoritativeTokens.size()) {
    result.matchesAuthoritativePath =
        std::equal(result.committedTokens.begin(), result.committedTokens.end(),
                   authoritativeTokens.begin(),
                   authoritativeTokens.begin() + result.committedTokens.size());
  }

  return result;
}

} // namespace us4
