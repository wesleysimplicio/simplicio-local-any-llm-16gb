#include <gtest/gtest.h>

#include "speculative/peagle_decoder.h"

namespace {

TEST(SpeculativeContractTest, DraftClampsToConfiguredWindow) {
  const us4::PEagleDecoder decoder(3U);
  const us4::PEagleDraft draft = decoder.Draft({10, 11, 12, 13, 14});

  ASSERT_EQ(draft.tokens.size(), 3U);
  EXPECT_EQ(draft.tokens[0], 10);
  EXPECT_EQ(draft.tokens[2], 12);
}

TEST(SpeculativeContractTest, VerifyMarksAllAcceptedWhenDraftMatchesTarget) {
  const us4::PEagleDecoder decoder(4U);
  const us4::PEagleDraft draft = decoder.Draft({7, 8, 9});
  const us4::PEagleVerificationResult result =
      decoder.Verify({7, 8, 9, 10}, draft);

  EXPECT_EQ(result.acceptedCount, 3U);
  EXPECT_EQ(result.rejectedCount, 0U);
  EXPECT_TRUE(result.allAccepted);
  EXPECT_TRUE(result.matchesAuthoritativePath);
  EXPECT_DOUBLE_EQ(result.acceptanceRate, 1.0);
  EXPECT_FALSE(result.fallbackToken.has_value());
  EXPECT_EQ(result.committedTokens, std::vector<int>({7, 8, 9}));
}

TEST(SpeculativeContractTest,
     VerifyCommitsAcceptedPrefixAndFallbackOnMismatch) {
  const us4::PEagleDecoder decoder(4U);
  const us4::PEagleDraft draft = decoder.Draft({4, 5, 6, 7});
  const us4::PEagleVerificationResult result =
      decoder.Verify({4, 5, 42, 8}, draft);

  EXPECT_EQ(result.acceptedCount, 2U);
  EXPECT_EQ(result.rejectedCount, 2U);
  EXPECT_FALSE(result.allAccepted);
  ASSERT_TRUE(result.fallbackToken.has_value());
  EXPECT_EQ(*result.fallbackToken, 42);
  EXPECT_TRUE(result.matchesAuthoritativePath);
  EXPECT_DOUBLE_EQ(result.acceptanceRate, 0.5);
  EXPECT_EQ(result.committedTokens, std::vector<int>({4, 5, 42}));
}

TEST(SpeculativeContractTest, EmptyDraftIsTriviallyEquivalent) {
  const us4::PEagleDecoder decoder(2U);
  const us4::PEagleVerificationResult result =
      decoder.Verify({1, 2, 3}, us4::PEagleDraft{});

  EXPECT_TRUE(result.committedTokens.empty());
  EXPECT_EQ(result.acceptedCount, 0U);
  EXPECT_EQ(result.rejectedCount, 0U);
  EXPECT_TRUE(result.allAccepted);
  EXPECT_TRUE(result.matchesAuthoritativePath);
  EXPECT_DOUBLE_EQ(result.acceptanceRate, 0.0);
}

} // namespace
