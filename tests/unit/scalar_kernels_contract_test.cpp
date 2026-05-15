#include <gtest/gtest.h>

#include <string>

#include "core/tensor.h"
#include "cpu/scalar_attention.h"
#include "cpu/scalar_matmul.h"

namespace {

void Fill(us4::Tensor& tensor, const std::initializer_list<float> values) {
  float* data = tensor.MutableDataAsFloat32();
  std::size_t index = 0;
  for (const float value : values) {
    data[index++] = value;
  }
}

}  // namespace

TEST(ScalarKernelsContractTest, MatmulMatchesExpectedReference) {
  us4::Tensor lhs({2, 3}, us4::DType::kFloat32);
  us4::Tensor rhs({3, 2}, us4::DType::kFloat32);
  us4::Tensor out({2, 2}, us4::DType::kFloat32);
  Fill(lhs, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  Fill(rhs, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

  std::string error;
  ASSERT_TRUE(us4::ScalarMatmul(lhs, rhs, out, &error)) << error;

  const float* data = out.DataAsFloat32();
  EXPECT_FLOAT_EQ(data[0], 58.0F);
  EXPECT_FLOAT_EQ(data[1], 64.0F);
  EXPECT_FLOAT_EQ(data[2], 139.0F);
  EXPECT_FLOAT_EQ(data[3], 154.0F);
}

TEST(ScalarKernelsContractTest, AttentionProducesStableWeightedOutput) {
  us4::Tensor query({1, 2}, us4::DType::kFloat32);
  us4::Tensor key({2, 2}, us4::DType::kFloat32);
  us4::Tensor value({2, 2}, us4::DType::kFloat32);
  us4::Tensor out({1, 2}, us4::DType::kFloat32);
  Fill(query, {1.0F, 0.0F});
  Fill(key, {1.0F, 0.0F, 0.0F, 1.0F});
  Fill(value, {2.0F, 0.0F, 0.0F, 4.0F});

  std::string error;
  ASSERT_TRUE(us4::ScalarAttention(query, key, value, out, false, {}, &error)) << error;

  const float* data = out.DataAsFloat32();
  EXPECT_GT(data[0], data[1]);
  EXPECT_GT(data[0], 0.0F);
  EXPECT_GT(data[1], 0.0F);
}
