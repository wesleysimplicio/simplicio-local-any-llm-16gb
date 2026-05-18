#include <gtest/gtest.h>

#include "speculative/eagle3_decoder.h"

namespace {

TEST(Eagle3ContractTest, BuildTreeClampsBranchBreadthAndDepth) {
  const us4::Eagle3Decoder decoder(2U, 3U);
  const us4::Eagle3DraftTree tree =
      decoder.BuildTree({{1, 2, 3, 4}, {9, 8, 7, 6}, {5, 5, 5, 5}});

  ASSERT_EQ(tree.branches.size(), 2U);
  EXPECT_EQ(tree.branches[0].tokens, std::vector<int>({1, 2, 3}));
  EXPECT_EQ(tree.branches[1].tokens, std::vector<int>({9, 8, 7}));
}

TEST(Eagle3ContractTest, VerifyChoosesBranchWithLongestSharedPrefix) {
  const us4::Eagle3Decoder decoder(3U, 4U);
  const us4::Eagle3DraftTree tree =
      decoder.BuildTree({{4, 5, 6, 7}, {4, 5, 42, 8}, {1, 2, 3, 4}});
  const us4::Eagle3VerificationResult result =
      decoder.Verify({4, 5, 42, 9}, tree);

  EXPECT_EQ(result.chosenBranchIndex, 1U);
  EXPECT_EQ(result.acceptedDepth, 3U);
  EXPECT_EQ(result.rejectedBranches, 2U);
  EXPECT_TRUE(result.foundMatchingBranch);
  EXPECT_TRUE(result.matchesAuthoritativePath);
  EXPECT_EQ(result.committedTokens, std::vector<int>({4, 5, 42, 9}));
  ASSERT_TRUE(result.fallbackToken.has_value());
  EXPECT_EQ(*result.fallbackToken, 9);
}

TEST(Eagle3ContractTest, VerifyHandlesFullBranchAcceptance) {
  const us4::Eagle3Decoder decoder(2U, 4U);
  const us4::Eagle3DraftTree tree = decoder.BuildTree({{7, 8, 9}, {7, 6, 5}});
  const us4::Eagle3VerificationResult result =
      decoder.Verify({7, 8, 9, 10}, tree);

  EXPECT_EQ(result.chosenBranchIndex, 0U);
  EXPECT_EQ(result.acceptedDepth, 3U);
  EXPECT_FALSE(result.fallbackToken.has_value());
  EXPECT_TRUE(result.matchesAuthoritativePath);
  EXPECT_EQ(result.committedTokens, std::vector<int>({7, 8, 9}));
}

TEST(Eagle3ContractTest, EmptyTreeProducesNoCommittedTokens) {
  const us4::Eagle3Decoder decoder(2U, 2U);
  const us4::Eagle3VerificationResult result =
      decoder.Verify({1, 2, 3}, us4::Eagle3DraftTree{});

  EXPECT_EQ(result.acceptedDepth, 0U);
  EXPECT_EQ(result.rejectedBranches, 0U);
  EXPECT_TRUE(result.committedTokens.empty());
  EXPECT_FALSE(result.fallbackToken.has_value());
  EXPECT_FALSE(result.matchesAuthoritativePath);
}

} // namespace
