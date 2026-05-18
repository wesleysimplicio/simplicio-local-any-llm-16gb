#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace us4 {

struct Eagle3Branch {
  std::vector<int> tokens;
};

struct Eagle3DraftTree {
  std::vector<Eagle3Branch> branches;
};

struct Eagle3VerificationResult {
  std::size_t chosenBranchIndex = 0U;
  std::size_t acceptedDepth = 0U;
  std::size_t rejectedBranches = 0U;
  std::vector<int> committedTokens;
  std::optional<int> fallbackToken;
  bool foundMatchingBranch = false;
  bool matchesAuthoritativePath = false;
};

class Eagle3Decoder {
public:
  Eagle3Decoder(std::size_t maxBranches = 3U, std::size_t maxDepth = 4U);

  [[nodiscard]] Eagle3DraftTree
  BuildTree(const std::vector<std::vector<int>> &proposalBranches) const;
  [[nodiscard]] Eagle3VerificationResult
  Verify(const std::vector<int> &authoritativeTokens,
         const Eagle3DraftTree &tree) const;

private:
  std::size_t maxBranches_ = 3U;
  std::size_t maxDepth_ = 4U;
};

} // namespace us4
