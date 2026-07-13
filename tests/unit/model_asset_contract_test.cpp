#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

#include "adapters/llama/llama_config.h"
#include "core/model_asset.h"

namespace {

struct ManifestExpectation {
  const char *directory;
  const char *family;
  const char *modelName;
  us4::DType weightDType;
  std::uint32_t seed;
  const char *defaultPromptToken;
  bool loadViaDirectory;
};

struct FileDetectionExpectation {
  const char *directory;
  const char *fileName;
  us4::ModelFormat format;
  const char *family;
  const char *modelName;
  us4::DType weightDType;
};

std::filesystem::path FixtureRoot() {
  return std::filesystem::path(US4_SOURCE_DIR) / "tests" / "fixtures" /
         "models";
}

} // namespace

TEST(ModelAssetContractTest, LoadsFixtureManifestMetadataAcrossFamilies) {
  constexpr std::array<ManifestExpectation, 9> kManifestExpectations = {{
      {"qwen-0.5b", "qwen", "qwen-0.5b-fixture", us4::DType::kFloat16, 41051U,
       "hi", false},
      {"gemma-2b-it", "gemma", "gemma-2b-it-fixture", us4::DType::kBFloat16,
       22073U, "hello", false},
      {"llama-3.1-8b", "llama", "llama-3.1-8b-fixture", us4::DType::kFloat16,
       31800U, "hello", true},
      {"bitnet-b1.58-2b", "bitnet", "bitnet-b1.58-2b-fixture",
       us4::DType::kInt8, 15802U, "hi", true},
      {"pt-bitnet-ternary-2b", "ternary", "pt-bitnet-ternary-2b-fixture",
       us4::DType::kInt4, 30021U, "hi", true},
      {"deepseek-v2-lite", "deepseek", "deepseek-v2-lite-fixture",
       us4::DType::kBFloat16, 52002U, "hi", true},
      {"glm-5.1", "glm", "glm-5.1-fixture", us4::DType::kBFloat16, 51045U, "hi",
       true},
      {"kimi-k2-instruct", "kimi", "kimi-k2-instruct-fixture",
       us4::DType::kBFloat16, 62002U, "hi", true},
      {"minimax-m2", "minimax", "minimax-m2-fixture", us4::DType::kBFloat16,
       72027U, "hi", true},
  }};

  for (const ManifestExpectation &expectation : kManifestExpectations) {
    SCOPED_TRACE(expectation.directory);

    const std::filesystem::path inputPath =
        expectation.loadViaDirectory
            ? FixtureRoot() / expectation.directory
            : FixtureRoot() / expectation.directory / "model.us4manifest";

    us4::ModelAsset asset;
    std::string error;
    ASSERT_TRUE(us4::LoadModelAsset(inputPath, asset, &error)) << error;
    EXPECT_EQ(asset.family, expectation.family);
    EXPECT_EQ(asset.modelName, expectation.modelName);
    EXPECT_EQ(asset.format, us4::ModelFormat::kFixtureManifest);
    EXPECT_EQ(asset.weightDType, expectation.weightDType);
    EXPECT_EQ(asset.seed, expectation.seed);
    EXPECT_EQ(asset.defaultPromptToken, expectation.defaultPromptToken);
    EXPECT_FALSE(asset.vocabulary.empty());
    EXPECT_EQ(asset.sourcePath.filename(), "model.us4manifest");
  }
}

TEST(ModelAssetContractTest, DetectsSupportedBinaryModelFormatsAcrossFamilies) {
  constexpr std::array<FileDetectionExpectation, 9> kFileDetectionExpectations =
      {{
          {"qwen-0.5b", "toy-qwen.gguf", us4::ModelFormat::kGguf, "qwen",
           "toy-qwen", us4::DType::kFloat16},
          {"gemma-2b-it", "toy-gemma.safetensors",
           us4::ModelFormat::kSafetensors, "gemma", "toy-gemma",
           us4::DType::kBFloat16},
          {"llama-3.1-8b", "toy-llama.gguf", us4::ModelFormat::kGguf, "llama",
           "toy-llama", us4::DType::kFloat16},
          {"bitnet-b1.58-2b", "toy-bitnet.gguf", us4::ModelFormat::kGguf,
           "bitnet", "toy-bitnet", us4::DType::kInt8},
          {"pt-bitnet-ternary-2b", "toy-ternary.safetensors",
           us4::ModelFormat::kSafetensors, "ternary", "toy-ternary",
           us4::DType::kInt4},
          {"deepseek-v2-lite", "toy-deepseek.safetensors",
           us4::ModelFormat::kSafetensors, "deepseek", "toy-deepseek",
           us4::DType::kBFloat16},
          {"glm-5.1", "toy-glm.safetensors", us4::ModelFormat::kSafetensors,
           "glm", "toy-glm", us4::DType::kBFloat16},
          {"kimi-k2-instruct", "toy-kimi.safetensors",
           us4::ModelFormat::kSafetensors, "kimi", "toy-kimi",
           us4::DType::kBFloat16},
          {"minimax-m2", "toy-minimax.safetensors",
           us4::ModelFormat::kSafetensors, "minimax", "toy-minimax",
           us4::DType::kBFloat16},
      }};

  for (const FileDetectionExpectation &expectation :
       kFileDetectionExpectations) {
    SCOPED_TRACE(expectation.fileName);

    const std::filesystem::path inputPath =
        FixtureRoot() / expectation.directory / expectation.fileName;

    us4::ModelAsset asset;
    std::string error;
    ASSERT_TRUE(us4::LoadModelAsset(inputPath, asset, &error)) << error;
    EXPECT_EQ(asset.format, expectation.format);
    EXPECT_EQ(asset.family, expectation.family);
    EXPECT_EQ(asset.modelName, expectation.modelName);
    EXPECT_EQ(asset.weightDType, expectation.weightDType);
    EXPECT_EQ(asset.sourcePath, inputPath);
  }
}

TEST(ModelAssetContractTest, LlamaManifestSurfacesTypedConfigMetadata) {
  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path inputPath =
      FixtureRoot() / "llama-3.1-8b" / "model.us4manifest";

  ASSERT_TRUE(us4::LoadModelAsset(inputPath, asset, &error)) << error;
  const us4::LlamaConfig config = us4::ResolveLlamaConfig(&asset);

  EXPECT_EQ(asset.metadata.at("hidden_size"), "8");
  EXPECT_EQ(asset.metadata.at("query_heads"), "2");
  EXPECT_EQ(asset.metadata.at("kv_heads"), "1");
  EXPECT_EQ(asset.metadata.at("head_dim"), "4");
  EXPECT_EQ(asset.metadata.at("rope_theta"), "10000");
  EXPECT_EQ(asset.metadata.at("rope_scaling"), "dynamic");
  EXPECT_EQ(asset.metadata.at("rope_scale"), "1.0");

  EXPECT_EQ(config.hiddenSize, 8U);
  EXPECT_EQ(config.queryHeads, 2U);
  EXPECT_EQ(config.kvHeads, 1U);
  EXPECT_EQ(config.headDim, 4U);
  EXPECT_FLOAT_EQ(config.ropeTheta, 10000.0F);
  EXPECT_EQ(config.ropeScaling, us4::RopeScalingType::kDynamic);
  EXPECT_FLOAT_EQ(config.ropeScale, 1.0F);
}

// Issue #81.2b: LoadModelAsset must actually open and parse a real .gguf
// container's tensor bytes, not just note the extension and hydrate from a
// sibling manifest. This fixture is a real GGUF file (see
// tests/fixtures/gguf/generate_toy_gguf.py), not the text placeholder the
// other .gguf fixtures above still use.
TEST(ModelAssetContractTest, RealGgufAssetExposesRealTensorBytes) {
  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path inputPath =
      std::filesystem::path(US4_SOURCE_DIR) / "tests" / "fixtures" / "gguf" /
      "toy_real.gguf";

  ASSERT_TRUE(us4::LoadModelAsset(inputPath, asset, &error)) << error;

  EXPECT_EQ(asset.format, us4::ModelFormat::kGguf);
  ASSERT_TRUE(asset.hasRealWeights);
  ASSERT_TRUE(asset.realTensors.contains("embedding.weight"));
  ASSERT_TRUE(asset.realTensors.contains("lm_head.weight"));
  EXPECT_EQ(asset.realTensorShapes.at("embedding.weight"),
            (std::vector<std::size_t>{4, 3}));
  EXPECT_EQ(asset.realTensorShapes.at("lm_head.weight"),
            (std::vector<std::size_t>{3, 4}));
  const std::vector<float> &embedding =
      asset.realTensors.at("embedding.weight");
  ASSERT_EQ(embedding.size(), 12U);
  EXPECT_FLOAT_EQ(embedding[0], 0.1F);
  EXPECT_FLOAT_EQ(embedding[11], 3.3F);
}

TEST(ModelAssetContractTest,
     LlamaBinaryAssetsInheritSiblingManifestMetadataWhenAvailable) {
  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path inputPath =
      FixtureRoot() / "llama-3.1-8b" / "toy-llama.gguf";

  ASSERT_TRUE(us4::LoadModelAsset(inputPath, asset, &error)) << error;

  EXPECT_EQ(asset.format, us4::ModelFormat::kGguf);
  EXPECT_EQ(asset.family, "llama");
  EXPECT_EQ(asset.modelName, "toy-llama");
  EXPECT_EQ(asset.weightDType, us4::DType::kFloat16);
  EXPECT_EQ(asset.seed, 31800U);
  EXPECT_EQ(asset.defaultPromptToken, "hello");
  EXPECT_FALSE(asset.vocabulary.empty());
  EXPECT_EQ(asset.metadata.at("hidden_size"), "8");
  EXPECT_EQ(asset.metadata.at("query_heads"), "2");
  EXPECT_EQ(asset.metadata.at("kv_heads"), "1");
  EXPECT_EQ(asset.metadata.at("tokenizer_json"),
            (FixtureRoot() / "llama-3.1-8b" / "tokenizer.json").string());
  EXPECT_EQ(asset.draftModelPath,
            (FixtureRoot() / "llama-3.1-8b" / "draft-llama.gguf"));
  EXPECT_EQ(asset.draftModelFormat, us4::ModelFormat::kGguf);
  EXPECT_TRUE(asset.sharedTokenizer);
}

TEST(ModelAssetContractTest,
     LowBitBinaryAssetsInheritSiblingManifestMetadataWhenAvailable) {
  const std::array<std::filesystem::path, 2> kPaths = {
      FixtureRoot() / "bitnet-b1.58-2b" / "toy-bitnet.gguf",
      FixtureRoot() / "pt-bitnet-ternary-2b" / "toy-ternary.safetensors",
  };

  for (const std::filesystem::path &inputPath : kPaths) {
    SCOPED_TRACE(inputPath.string());

    us4::ModelAsset asset;
    std::string error;
    ASSERT_TRUE(us4::LoadModelAsset(inputPath, asset, &error)) << error;

    EXPECT_FALSE(asset.vocabulary.empty());
    EXPECT_EQ(asset.defaultPromptToken, "hi");
    EXPECT_TRUE(asset.metadata.contains("weight_dtype"));
    EXPECT_TRUE(asset.metadata.contains("tokenizer_json"));
  }
}

TEST(ModelAssetContractTest,
     MoeAssetsSurfaceShardAwareLoaderMetadataAcrossManifestAndBinaryInputs) {
  const std::array<std::filesystem::path, 8> kInputs = {
      FixtureRoot() / "deepseek-v2-lite" / "model.us4manifest",
      FixtureRoot() / "deepseek-v2-lite" / "toy-deepseek.safetensors",
      FixtureRoot() / "glm-5.1" / "model.us4manifest",
      FixtureRoot() / "glm-5.1" / "toy-glm.safetensors",
      FixtureRoot() / "kimi-k2-instruct" / "model.us4manifest",
      FixtureRoot() / "kimi-k2-instruct" / "toy-kimi.safetensors",
      FixtureRoot() / "minimax-m2" / "model.us4manifest",
      FixtureRoot() / "minimax-m2" / "toy-minimax.safetensors",
  };

  for (const std::filesystem::path &inputPath : kInputs) {
    SCOPED_TRACE(inputPath.string());

    us4::ModelAsset asset;
    std::string error;
    ASSERT_TRUE(us4::LoadModelAsset(inputPath, asset, &error)) << error;

    EXPECT_TRUE(asset.moeLazyLoad);
    EXPECT_EQ(asset.moeActiveExperts, 2U);
    ASSERT_EQ(asset.expertShardPaths.size(), 2U);
    EXPECT_EQ(asset.expertShardPaths[0].filename(), "experts-00.safetensors");
    EXPECT_EQ(asset.expertShardPaths[1].filename(), "experts-01.safetensors");
    EXPECT_EQ(asset.metadata.at("moe_lazy_load"), "true");
    EXPECT_EQ(asset.metadata.at("moe_active_experts"), "2");
    EXPECT_EQ(asset.metadata.at("moe_expert_shards"),
              "experts-00.safetensors,experts-01.safetensors");
  }
}

TEST(ModelAssetContractTest,
     FrontierMoeAssetsExposeFamilyTokenizerAndChatTemplateMetadata) {
  const std::array<std::filesystem::path, 6> kInputs = {
      FixtureRoot() / "deepseek-v2-lite" / "model.us4manifest",
      FixtureRoot() / "deepseek-v2-lite" / "toy-deepseek.safetensors",
      FixtureRoot() / "glm-5.1" / "model.us4manifest",
      FixtureRoot() / "glm-5.1" / "toy-glm.safetensors",
      FixtureRoot() / "kimi-k2-instruct" / "model.us4manifest",
      FixtureRoot() / "kimi-k2-instruct" / "toy-kimi.safetensors",
  };

  for (const std::filesystem::path &inputPath : kInputs) {
    SCOPED_TRACE(inputPath.string());

    us4::ModelAsset asset;
    std::string error;
    ASSERT_TRUE(us4::LoadModelAsset(inputPath, asset, &error)) << error;

    ASSERT_TRUE(asset.metadata.contains("tokenizer_json"));
    EXPECT_FALSE(asset.metadata.at("tokenizer_json").empty());
    ASSERT_TRUE(asset.metadata.contains("chat_template"));
    EXPECT_FALSE(asset.metadata.at("chat_template").empty());
    EXPECT_EQ(asset.chatTemplate, asset.metadata.at("chat_template"));
  }
}
