#include "speculative/eagle3_decoder.h"

#include <algorithm>

namespace us4 {

namespace {

std::size_t SharedPrefix(const std::vector<int> &lhs,
                         const std::vector<int> &rhs) {
  const std::size_t limit = std::min(lhs.size(), rhs.size());
  std::size_t count = 0U;
  while (count < limit && lhs[count] == rhs[count]) {
    ++count;
  }
  return count;
}

} // namespace

Eagle3Decoder::Eagle3Decoder(const std::size_t maxBranches,
                             const std::size_t maxDepth)
    : maxBranches_(std::max<std::size_t>(1U, maxBranches)),
      maxDepth_(std::max<std::size_t>(1U, maxDepth)) {}

Eagle3DraftTree Eagle3Decoder::BuildTree(
    const std::vector<std::vector<int>> &proposalBranches) const {
  Eagle3DraftTree tree;
  const std::size_t branchCount =
      std::min<std::size_t>(maxBranches_, proposalBranches.size());
  tree.branches.reserve(branchCount);
  for (std::size_t index = 0; index < branchCount; ++index) {
    Eagle3Branch branch;
    const std::vector<int> &proposal = proposalBranches[index];
    const std::size_t depth = std::min<std::size_t>(maxDepth_, proposal.size());
    branch.tokens.assign(proposal.begin(), proposal.begin() + depth);
    tree.branches.push_back(std::move(branch));
  }
  return tree;
}

Eagle3VerificationResult
Eagle3Decoder::Verify(const std::vector<int> &authoritativeTokens,
                      const Eagle3DraftTree &tree) const {
  Eagle3VerificationResult result;
  if (tree.branches.empty()) {
    result.matchesAuthoritativePath = authoritativeTokens.empty();
    return result;
  }

  std::size_t bestIndex = 0U;
  std::size_t bestPrefix = 0U;
  for (std::size_t index = 0; index < tree.branches.size(); ++index) {
    const std::size_t prefix =
        SharedPrefix(tree.branches[index].tokens, authoritativeTokens);
    if (prefix > bestPrefix) {
      bestPrefix = prefix;
      bestIndex = index;
    }
  }

  result.chosenBranchIndex = bestIndex;
  result.acceptedDepth = bestPrefix;
  result.rejectedBranches = tree.branches.size() - 1U;
  result.foundMatchingBranch = bestPrefix > 0U;

  const Eagle3Branch &branch = tree.branches[bestIndex];
  result.committedTokens.assign(branch.tokens.begin(),
                                branch.tokens.begin() + bestPrefix);

  if (bestPrefix < authoritativeTokens.size() &&
      bestPrefix < branch.tokens.size()) {
    result.fallbackToken = authoritativeTokens[bestPrefix];
    result.committedTokens.push_back(*result.fallbackToken);
  }

  if (result.committedTokens.size() <= authoritativeTokens.size()) {
    result.matchesAuthoritativePath =
        std::equal(result.committedTokens.begin(), result.committedTokens.end(),
                   authoritativeTokens.begin(),
                   authoritativeTokens.begin() + result.committedTokens.size());
  }

  return result;
}

} // namespace us4
