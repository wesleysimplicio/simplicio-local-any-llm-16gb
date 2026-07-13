#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "core/bpe_tokenizer.h"
#include "core/json_value.h"

namespace us4 {
namespace {

std::filesystem::path FixtureDir() {
  return std::filesystem::path(US4_SOURCE_DIR) / "tests" / "fixtures" /
         "tokenizer";
}

JsonValue LoadJsonFixture(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return JsonValue::Parse(buffer.str());
}

TEST(BpeTokenizerContractTest, RejectsPlaceholderManifestTokenizer) {
  const std::filesystem::path placeholder =
      std::filesystem::path(US4_SOURCE_DIR) / "tests" / "fixtures" / "models" /
      "llama-3.1-8b" / "tokenizer.json";
  std::string error;
  const auto tokenizer = BpeTokenizer::LoadFromFile(placeholder, &error);
  EXPECT_FALSE(tokenizer.has_value());
  EXPECT_FALSE(error.empty());
}

TEST(BpeTokenizerContractTest, LoadsRealVocabAndMerges) {
  std::string error;
  const auto tokenizer = BpeTokenizer::LoadFromFile(
      FixtureDir() / "toy_bpe_tokenizer.json", &error);
  ASSERT_TRUE(tokenizer.has_value()) << error;
  EXPECT_GT(tokenizer->VocabSize(), 0U);
  EXPECT_TRUE(tokenizer->HasToken("low"));
  EXPECT_TRUE(tokenizer->HasToken("newer"));
}

TEST(BpeTokenizerContractTest, MatchesIndependentOracleTokenIds) {
  std::string error;
  const auto tokenizer = BpeTokenizer::LoadFromFile(
      FixtureDir() / "toy_bpe_tokenizer.json", &error);
  ASSERT_TRUE(tokenizer.has_value()) << error;

  const JsonValue reference =
      LoadJsonFixture(FixtureDir() / "reference_output.json");
  ASSERT_TRUE(reference.IsObject());
  ASSERT_GT(reference.Entries().size(), 0U);

  for (const auto &[sentence, expectedIdsJson] : reference.Entries()) {
    ASSERT_TRUE(expectedIdsJson.IsArray());
    std::vector<int> expectedIds;
    for (const JsonValue &idValue : expectedIdsJson.AsArray()) {
      expectedIds.push_back(static_cast<int>(idValue.AsNumber()));
    }

    const std::vector<int> actualIds = tokenizer->EncodeIds(sentence);
    EXPECT_EQ(actualIds, expectedIds) << "mismatch for sentence: " << sentence;
  }
}

} // namespace
} // namespace us4
