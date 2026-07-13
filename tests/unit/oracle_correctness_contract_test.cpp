#include <filesystem>
#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "core/json_value.h"
#include "core/safetensors_reader.h"
#include "core/tensor.h"
#include "cpu/scalar_matmul.h"

namespace us4 {
namespace {

// Correctness contract per issue #82: the native forward path must be
// checked against an EXTERNAL oracle (an independently written Python
// implementation over the same real weights), not against the runtime's
// own scalar kernel. See
// tests/fixtures/safetensors/generate_oracle_reference.py.
constexpr float kToleranceAbs = 1e-4F;

std::filesystem::path FixtureDir() {
  return std::filesystem::path(US4_SOURCE_DIR) / "tests" / "fixtures" /
         "safetensors";
}

JsonValue LoadJsonFixture(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return JsonValue::Parse(buffer.str());
}

TEST(OracleCorrectnessContractTest, FailsExplicitlyWhenWeightsMissing) {
  std::string error;
  const auto reader = SafetensorsReader::Open(
      FixtureDir() / "does_not_exist.safetensors", &error);
  ASSERT_FALSE(reader.has_value());
  EXPECT_FALSE(error.empty());
}

TEST(OracleCorrectnessContractTest,
     NativeForwardMatchesIndependentOracleLogits) {
  std::string error;
  const auto reader =
      SafetensorsReader::Open(FixtureDir() / "toy_real.safetensors", &error);
  ASSERT_TRUE(reader.has_value()) << error;

  const auto *embeddingInfo = reader->Find("embedding.weight");
  const auto *lmHeadInfo = reader->Find("lm_head.weight");
  ASSERT_NE(embeddingInfo, nullptr);
  ASSERT_NE(lmHeadInfo, nullptr);

  const std::vector<float> embedding =
      reader->ReadFloat32("embedding.weight", &error);
  ASSERT_FALSE(embedding.empty()) << error;
  const std::vector<float> lmHead = reader->ReadFloat32("lm_head.weight");
  ASSERT_FALSE(lmHead.empty());

  const std::size_t vocabSize = embeddingInfo->shape[0];
  const std::size_t hiddenSize = embeddingInfo->shape[1];
  const std::size_t outputSize = lmHeadInfo->shape[1];

  const JsonValue reference =
      LoadJsonFixture(FixtureDir() / "oracle_reference_logits.json");
  ASSERT_TRUE(reference.IsObject());
  ASSERT_EQ(reference.Entries().size(), vocabSize);

  Tensor lmHeadTensor({hiddenSize, outputSize}, DType::kFloat32);
  std::copy(lmHead.begin(), lmHead.end(), lmHeadTensor.MutableDataAsFloat32());

  for (std::size_t tokenId = 0; tokenId < vocabSize; ++tokenId) {
    Tensor rowTensor({1, hiddenSize}, DType::kFloat32);
    std::copy(embedding.begin() +
                  static_cast<std::ptrdiff_t>(tokenId * hiddenSize),
              embedding.begin() +
                  static_cast<std::ptrdiff_t>((tokenId + 1) * hiddenSize),
              rowTensor.MutableDataAsFloat32());

    Tensor logitsTensor({1, outputSize}, DType::kFloat32);
    ASSERT_TRUE(ScalarMatmul(rowTensor, lmHeadTensor, logitsTensor, &error))
        << error;

    const JsonValue &expectedRow = reference[std::to_string(tokenId)];
    ASSERT_TRUE(expectedRow.IsArray());
    ASSERT_EQ(expectedRow.AsArray().size(), outputSize);

    const float *actualLogits = logitsTensor.DataAsFloat32();
    for (std::size_t col = 0; col < outputSize; ++col) {
      EXPECT_NEAR(actualLogits[col], expectedRow.AsArray()[col].AsNumber(),
                  kToleranceAbs)
          << "token " << tokenId << " column " << col;
    }
  }
}

} // namespace
} // namespace us4
