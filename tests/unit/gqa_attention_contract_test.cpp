#include <gtest/gtest.h>

#include <string>

#include "core/gqa_attention.h"
#include "core/tensor.h"
#include "cpu/scalar_attention.h"

namespace {

void Fill(us4::Tensor &tensor, const std::initializer_list<float> values) {
  float *data = tensor.MutableDataAsFloat32();
  std::size_t index = 0;
  for (const float value : values) {
    data[index++] = value;
  }
}

} // namespace

TEST(GqaAttentionContractTest, SingleQueryHeadMatchesScalarAttention) {
  us4::Tensor query({1, 2}, us4::DType::kFloat32);
  us4::Tensor key({2, 2}, us4::DType::kFloat32);
  us4::Tensor value({2, 2}, us4::DType::kFloat32);
  us4::Tensor scalarOut({1, 2}, us4::DType::kFloat32);
  us4::Tensor gqaOut({1, 2}, us4::DType::kFloat32);
  Fill(query, {1.0F, 0.0F});
  Fill(key, {1.0F, 0.0F, 0.0F, 1.0F});
  Fill(value, {2.0F, 0.0F, 0.0F, 4.0F});

  std::string error;
  ASSERT_TRUE(
      us4::ScalarAttention(query, key, value, scalarOut, false, {}, &error))
      << error;
  ASSERT_TRUE(us4::GqaAttention(query, key, value, 1U, 1U, gqaOut, &error))
      << error;

  const float *scalarData = scalarOut.DataAsFloat32();
  const float *gqaData = gqaOut.DataAsFloat32();
  ASSERT_NE(scalarData, nullptr);
  ASSERT_NE(gqaData, nullptr);
  EXPECT_NEAR(gqaData[0], scalarData[0], 1e-5F);
  EXPECT_NEAR(gqaData[1], scalarData[1], 1e-5F);
}

TEST(GqaAttentionContractTest, GroupedQueryHeadsProduceStableHeadWiseOutput) {
  us4::Tensor query({1, 4}, us4::DType::kFloat32);
  us4::Tensor key({2, 2}, us4::DType::kFloat32);
  us4::Tensor value({2, 2}, us4::DType::kFloat32);
  us4::Tensor out({1, 4}, us4::DType::kFloat32);
  Fill(query, {1.0F, 0.0F, 0.0F, 1.0F});
  Fill(key, {1.0F, 0.0F, 0.0F, 1.0F});
  Fill(value, {10.0F, 0.0F, 0.0F, 20.0F});

  std::string error;
  ASSERT_TRUE(us4::GqaAttention(query, key, value, 2U, 1U, out, &error))
      << error;

  const float *data = out.DataAsFloat32();
  ASSERT_NE(data, nullptr);
  EXPECT_NEAR(data[0], 6.69762F, 1e-4F);
  EXPECT_NEAR(data[1], 6.60478F, 1e-4F);
  EXPECT_NEAR(data[2], 3.30238F, 1e-4F);
  EXPECT_NEAR(data[3], 13.3952F, 1e-4F);
}

TEST(GqaAttentionContractTest, InvalidHeadTopologyFailsDeterministically) {
  us4::Tensor query({1, 4}, us4::DType::kFloat32);
  us4::Tensor key({2, 2}, us4::DType::kFloat32);
  us4::Tensor value({2, 2}, us4::DType::kFloat32);
  us4::Tensor out({1, 4}, us4::DType::kFloat32);

  std::string error;
  EXPECT_FALSE(us4::GqaAttention(query, key, value, 3U, 2U, out, &error));
  EXPECT_EQ(error, "invalid GQA head relationship");
}
