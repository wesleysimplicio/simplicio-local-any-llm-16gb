#include <filesystem>

#include <gtest/gtest.h>

#include "core/safetensors_reader.h"

namespace us4 {
namespace {

std::filesystem::path FixtureDir() {
  return std::filesystem::path(US4_SOURCE_DIR) / "tests" / "fixtures" /
         "safetensors";
}

TEST(SafetensorsReaderContractTest, RejectsPlaceholderTextFile) {
  const std::filesystem::path placeholder =
      std::filesystem::path(US4_SOURCE_DIR) / "tests" / "fixtures" / "models" /
      "gemma-2b-it" / "toy-gemma.safetensors";
  std::string error;
  const auto reader = SafetensorsReader::Open(placeholder, &error);
  EXPECT_FALSE(reader.has_value());
  EXPECT_FALSE(error.empty());
}

TEST(SafetensorsReaderContractTest, ParsesRealHeaderAndShapes) {
  std::string error;
  const auto reader =
      SafetensorsReader::Open(FixtureDir() / "toy_real.safetensors", &error);
  ASSERT_TRUE(reader.has_value()) << error;
  EXPECT_EQ(reader->TensorCount(), 2U);

  const auto *embedding = reader->Find("embedding.weight");
  ASSERT_NE(embedding, nullptr);
  EXPECT_EQ(embedding->dtype, "F32");
  EXPECT_EQ(embedding->shape, (std::vector<std::size_t>{4, 3}));

  const auto *lmHead = reader->Find("lm_head.weight");
  ASSERT_NE(lmHead, nullptr);
  EXPECT_EQ(lmHead->shape, (std::vector<std::size_t>{3, 4}));
}

TEST(SafetensorsReaderContractTest, ReadsRealFloat32TensorBytes) {
  std::string error;
  const auto reader =
      SafetensorsReader::Open(FixtureDir() / "toy_real.safetensors", &error);
  ASSERT_TRUE(reader.has_value()) << error;

  const std::vector<float> embedding =
      reader->ReadFloat32("embedding.weight", &error);
  ASSERT_EQ(embedding.size(), 12U) << error;
  const std::vector<float> expectedEmbedding = {
      0.1F, 0.2F, 0.3F, 1.1F, 1.2F, 1.3F, 2.1F, 2.2F, 2.3F, 3.1F, 3.2F, 3.3F,
  };
  for (std::size_t i = 0; i < expectedEmbedding.size(); ++i) {
    EXPECT_NEAR(embedding[i], expectedEmbedding[i], 1e-6F);
  }

  const std::vector<float> lmHead = reader->ReadFloat32("lm_head.weight");
  ASSERT_EQ(lmHead.size(), 12U);
  EXPECT_FLOAT_EQ(lmHead[0], 0.5F);
  EXPECT_FLOAT_EQ(lmHead[11], -3.0F);
}

// Issue #81.11: real production checkpoints (e.g. Qwen2.5-0.5B's
// model.safetensors, downloaded and inspected while validating this) store
// weights as BF16, not F32 -- ReadFloat32 used to reject that dtype
// outright. Values here are exact powers of two, so bf16 truncation loses
// no bits and the oracle can assert exact equality.
TEST(SafetensorsReaderContractTest, ReadsRealBf16TensorBytesAsFloat32) {
  std::string error;
  const auto reader =
      SafetensorsReader::Open(FixtureDir() / "toy_bf16.safetensors", &error);
  ASSERT_TRUE(reader.has_value()) << error;

  const auto *info = reader->Find("embedding.weight");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->dtype, "BF16");

  const std::vector<float> values =
      reader->ReadFloat32("embedding.weight", &error);
  const std::vector<float> expected = {1.0F, -1.0F, 2.0F, -2.0F,
                                       0.5F, -0.5F, 4.0F, -4.0F};
  ASSERT_EQ(values.size(), expected.size()) << error;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(values[i], expected[i]);
  }
}

TEST(SafetensorsReaderContractTest, MissingTensorReportsExplicitError) {
  const auto reader =
      SafetensorsReader::Open(FixtureDir() / "toy_real.safetensors");
  ASSERT_TRUE(reader.has_value());
  std::string error;
  const std::vector<float> missing =
      reader->ReadFloat32("does.not.exist", &error);
  EXPECT_TRUE(missing.empty());
  EXPECT_FALSE(error.empty());
}

} // namespace
} // namespace us4
