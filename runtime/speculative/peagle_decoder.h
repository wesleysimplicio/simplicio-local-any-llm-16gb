#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace us4 {

struct PEagleDraft {
  std::vector<int> tokens;
};

struct PEagleVerificationResult {
  std::vector<int> committedTokens;
  std::size_t acceptedCount = 0U;
  std::size_t rejectedCount = 0U;
  std::optional<int> fallbackToken;
  double acceptanceRate = 0.0;
  bool allAccepted = false;
  bool matchesAuthoritativePath = false;
};

class PEagleDecoder {
public:
  explicit PEagleDecoder(std::size_t maxDraftTokens = 4U);

  [[nodiscard]] std::size_t MaxDraftTokens() const noexcept;
  [[nodiscard]] PEagleDraft Draft(const std::vector<int> &proposalTokens) const;
  [[nodiscard]] PEagleVerificationResult
  Verify(const std::vector<int> &authoritativeTokens,
         const PEagleDraft &draft) const;

private:
  std::size_t maxDraftTokens_ = 4U;
};

} // namespace us4
