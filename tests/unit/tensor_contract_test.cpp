#include <gtest/gtest.h>

#include <vector>

#include "core/tensor.h"

TEST(TensorContractTest, ComputesShapeStrideAndStorageForFloat32) {
  const us4::Tensor tensor({2, 3, 4}, us4::DType::kFloat32);

  EXPECT_EQ(tensor.Rank(), 3U);
  EXPECT_EQ(tensor.ElementCount(), 24U);
  EXPECT_EQ(tensor.ByteSize(), 96U);
  EXPECT_EQ(tensor.Shape(), (std::vector<std::size_t>{2, 3, 4}));
  EXPECT_EQ(tensor.Strides(), (std::vector<std::size_t>{12, 4, 1}));
  EXPECT_TRUE(tensor.IsContiguous());
}

TEST(TensorContractTest, ReshapePreservesElementCount) {
  us4::Tensor tensor({2, 4}, us4::DType::kFloat32);

  EXPECT_TRUE(tensor.Reshape({1, 8}));
  EXPECT_FALSE(tensor.Reshape({3, 3}));
  EXPECT_EQ(tensor.Shape(), (std::vector<std::size_t>{1, 8}));
}

TEST(TensorContractTest, AccountsForPackedInt4Storage) {
  const us4::Tensor tensor({3, 3}, us4::DType::kInt4);
  EXPECT_EQ(tensor.ElementCount(), 9U);
  EXPECT_EQ(tensor.ByteSize(), 5U);
}
