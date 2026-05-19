#include <gtest/gtest.h>

#include <cstdint>

#include "cpu/ternary_lut.h"
#include "neon/bitnet_matmul.h"

TEST(LowBitContractTest, BitNetByteRoundTrips) {
  for (std::int8_t a : {-1, 0, 1}) {
    for (std::int8_t b : {-1, 0, 1}) {
      for (std::int8_t c : {-1, 0, 1}) {
        for (std::int8_t d : {-1, 0, 1}) {
          for (std::int8_t e : {-1, 0, 1}) {
            const std::int8_t input[5] = {a, b, c, d, e};
            std::uint8_t packed = 0;
            ASSERT_TRUE(us4::EncodeBitNetByte(input, packed));
            std::int8_t decoded[5] = {};
            us4::DecodeBitNetByte(packed, decoded);
            EXPECT_EQ(decoded[0], a);
            EXPECT_EQ(decoded[1], b);
            EXPECT_EQ(decoded[2], c);
            EXPECT_EQ(decoded[3], d);
            EXPECT_EQ(decoded[4], e);
          }
        }
      }
    }
  }
}

TEST(LowBitContractTest, BitNetEncodeRejectsOutOfRangeValues) {
  const std::int8_t invalid[5] = {2, 0, 0, 0, 0};
  std::uint8_t packed = 0;
  EXPECT_FALSE(us4::EncodeBitNetByte(invalid, packed));
}

TEST(LowBitContractTest, BitNetMatmulAppliesRowScale) {
  us4::BitNetPackedMatrix weights;
  weights.rows = 5;
  weights.cols = 2;
  weights.rowScale = {2.0F, 0.5F};

  const std::int8_t row0[5] = {1, 1, 0, -1, 0};
  const std::int8_t row1[5] = {0, 1, 1, 0, -1};
  std::uint8_t packedRow0 = 0;
  std::uint8_t packedRow1 = 0;
  ASSERT_TRUE(us4::EncodeBitNetByte(row0, packedRow0));
  ASSERT_TRUE(us4::EncodeBitNetByte(row1, packedRow1));
  weights.packed = {packedRow0, packedRow1};

  const std::vector<float> activations = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
  const auto output = us4::BitNetMatmul(activations, 1, weights);

  ASSERT_EQ(output.size(), 2U);
  // row0 dot ones = 1+1+0-1+0 = 1, then * 2.0 = 2.0
  EXPECT_FLOAT_EQ(output[0], 2.0F);
  // row1 dot ones = 0+1+1+0-1 = 1, then * 0.5 = 0.5
  EXPECT_FLOAT_EQ(output[1], 0.5F);
}

TEST(LowBitContractTest, TernaryLutCoversAllChunkValues) {
  const auto lut = us4::BuildTernaryLookupTable();
  EXPECT_EQ(lut.size(), us4::kTernaryLutSize);

  // Spot-check a couple of entries.
  const auto first = us4::DecodeTernaryChunk(0U);
  EXPECT_EQ(first.values[0], -1);
  EXPECT_EQ(first.values[1], -1);
  EXPECT_EQ(first.values[2], -1);
  EXPECT_EQ(first.values[3], -1);

  const auto last = us4::DecodeTernaryChunk(80U);
  EXPECT_EQ(last.values[0], 1);
  EXPECT_EQ(last.values[1], 1);
  EXPECT_EQ(last.values[2], 1);
  EXPECT_EQ(last.values[3], 1);
}

TEST(LowBitContractTest, TernaryChunkDotMatchesScalarReference) {
  const auto lut = us4::BuildTernaryLookupTable();
  us4::TernaryChunkValues values;
  values.values = {1, -1, 0, 1};
  std::uint8_t packed = 0;
  ASSERT_TRUE(us4::EncodeTernaryChunk(values, packed));

  const float activations[4] = {1.0F, 2.0F, 4.0F, 8.0F};
  const float result = us4::TernaryChunkDot(lut, packed, activations);
  EXPECT_FLOAT_EQ(result, 1.0F - 2.0F + 0.0F + 8.0F);
}

TEST(LowBitContractTest, TernaryChunkDotReturnsZeroForOutOfRange) {
  const auto lut = us4::BuildTernaryLookupTable();
  const float activations[4] = {1.0F, 1.0F, 1.0F, 1.0F};
  EXPECT_FLOAT_EQ(us4::TernaryChunkDot(lut, 200U, activations), 0.0F);
}
