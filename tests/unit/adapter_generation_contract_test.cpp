#include <filesystem>

#include <gtest/gtest.h>

#include "adapters/adapter_registry.h"
#include "core/hardware_probe.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"

namespace {

us4::HardwareProbeResult MakeProbe() {
  us4::HardwareProbeResult probe;
  probe.platform = "macos";
  probe.architecture = "arm64";
  probe.chip = "apple-m";
  probe.unifiedMemoryGiB = 24;
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasPerformanceCores = true;
  probe.hasEfficiencyCores = true;
  probe.recommendedMode = us4::RuntimeMode::kDegraded;
  return probe;
}

std::filesystem::path RepoRoot() {
#ifdef US4_SOURCE_DIR
  return std::filesystem::path(US4_SOURCE_DIR);
#else
  return std::filesystem::path(__FILE__)
      .parent_path()
      .parent_path()
      .parent_path();
#endif
}

} // namespace

TEST(AdapterGenerationContractTest,
     RegistryFindsAdaptersByModelAndFamilyAliases) {
  const us4::IUS4V6Adapter *qwen = us4::FindAdapterByModel("QWEN-0.5B");
  const us4::IUS4V6Adapter *gemma = us4::FindAdapterByModel("gemma");
  const us4::IUS4V6Adapter *deepseek = us4::FindAdapterByModel("deepseek");

  ASSERT_NE(qwen, nullptr);
  ASSERT_NE(gemma, nullptr);
  ASSERT_NE(deepseek, nullptr);
  EXPECT_EQ(qwen->Architecture(), us4::ArchitectureType::kDense);
  EXPECT_EQ(gemma->Architecture(), us4::ArchitectureType::kDense);
  EXPECT_EQ(deepseek->Architecture(), us4::ArchitectureType::kMoe);
}

TEST(AdapterGenerationContractTest,
     DenseAdapterUsesFixtureManifestAndRequestedBackendFallback) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("qwen-0.5b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path manifest = RepoRoot() / "tests" / "fixtures" /
                                         "models" / "qwen-0.5b" /
                                         "model.us4manifest";
  ASSERT_TRUE(us4::LoadModelAsset(manifest, asset, &error)) << error;

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result =
      adapter->Generate({.prompt = "Hi, US4!",
                         .maxTokens = 5,
                         .asset = &asset,
                         .requestedBackend = us4::BackendType::kMetal},
                        context);

  EXPECT_EQ(result.family, "qwen");
  EXPECT_EQ(result.modelName, "qwen-0.5b-fixture");
  EXPECT_EQ(result.assetFormat, "fixture-manifest");
  EXPECT_EQ(result.assetPath, manifest.string());
  EXPECT_EQ(result.mode, us4::RuntimeMode::kDegraded);
  EXPECT_EQ(result.backend, "neon");
  EXPECT_EQ(result.backendReason, "requested-backend-unavailable");
  EXPECT_EQ(result.weightDType, "fp16");
  EXPECT_EQ(result.neonKernelFlavor, "fp16-lane8");
  EXPECT_EQ(result.dequantPath, "none");
  EXPECT_FALSE(result.kvCacheHit);
  EXPECT_EQ(result.kvPageCount, 1U);
  EXPECT_EQ(result.kvHotPages, 1U);
  EXPECT_EQ(result.kvWarmPages, 0U);
  EXPECT_EQ(result.kvColdPages, 0U);
  EXPECT_EQ(result.prefixCacheEntries, 1U);
  EXPECT_TRUE(result.fellBack);
  EXPECT_EQ(result.promptTokens.size(), 4U);
  EXPECT_EQ(result.promptTokens[0], "hi");
  EXPECT_EQ(result.promptTokens[1], ",");
  EXPECT_EQ(result.promptTokens[2], "us4");
  EXPECT_EQ(result.promptTokens[3], "!");
  EXPECT_EQ(result.generatedTokens.size(), 5U);
  EXPECT_FALSE(result.text.empty());
}

TEST(AdapterGenerationContractTest,
     EmptyPromptFallsBackToFixtureDefaultPromptToken) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("gemma-2b-it");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path manifest = RepoRoot() / "tests" / "fixtures" /
                                         "models" / "gemma-2b-it" /
                                         "model.us4manifest";
  ASSERT_TRUE(us4::LoadModelAsset(manifest, asset, &error)) << error;

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result = adapter->Generate(
      {.prompt = "", .maxTokens = 3, .asset = &asset}, context);

  ASSERT_EQ(result.promptTokens.size(), 1U);
  EXPECT_EQ(result.promptTokens[0], "hello");
  EXPECT_EQ(result.family, "gemma");
  EXPECT_EQ(result.modelName, "gemma-2b-it-fixture");
  EXPECT_EQ(result.backend, "neon");
  EXPECT_EQ(result.backendReason, "auto");
  EXPECT_EQ(result.weightDType, "bf16");
  EXPECT_EQ(result.neonKernelFlavor, "bf16-lane8");
  EXPECT_EQ(result.dequantPath, "none");
  EXPECT_FALSE(result.kvCacheHit);
  EXPECT_FALSE(result.fellBack);
  EXPECT_EQ(result.generatedTokens.size(), 3U);
}

TEST(AdapterGenerationContractTest, LowBitAssetsSurfaceNeonDequantIntent) {
  const us4::IUS4V6Adapter *bitnet = us4::FindAdapterByModel("bitnet-b1.58-2b");
  const us4::IUS4V6Adapter *ternary =
      us4::FindAdapterByModel("pt-bitnet-ternary-2b");
  ASSERT_NE(bitnet, nullptr);
  ASSERT_NE(ternary, nullptr);

  us4::ModelAsset int8Asset;
  us4::ModelAsset int4Asset;
  std::string error;
  const std::filesystem::path int8Manifest = RepoRoot() / "tests" / "fixtures" /
                                             "models" / "bitnet-b1.58-2b" /
                                             "model.us4manifest";
  const std::filesystem::path int4Manifest = RepoRoot() / "tests" / "fixtures" /
                                             "models" / "pt-bitnet-ternary-2b" /
                                             "model.us4manifest";
  ASSERT_TRUE(us4::LoadModelAsset(int8Manifest, int8Asset, &error)) << error;
  ASSERT_TRUE(us4::LoadModelAsset(int4Manifest, int4Asset, &error)) << error;

  us4::RuntimeContext context(MakeProbe());
  bitnet->ConfigureRuntime(context);
  const us4::GenerationResult int8Result = bitnet->Generate(
      {.prompt = "hi", .maxTokens = 2, .asset = &int8Asset}, context);
  EXPECT_EQ(int8Result.backend, "neon");
  EXPECT_EQ(int8Result.weightDType, "int8");
  EXPECT_EQ(int8Result.dequantPath, "groupwise-int8");
  EXPECT_EQ(int8Result.neonKernelFlavor, "int8-dot");
  EXPECT_EQ(int8Result.kvPageCount, 1U);

  us4::RuntimeContext ternaryContext(MakeProbe());
  ternary->ConfigureRuntime(ternaryContext);
  const us4::GenerationResult int4Result = ternary->Generate(
      {.prompt = "hi", .maxTokens = 2, .asset = &int4Asset}, ternaryContext);
  EXPECT_EQ(int4Result.backend, "neon");
  EXPECT_EQ(int4Result.weightDType, "int4");
  EXPECT_EQ(int4Result.dequantPath, "groupwise-int4");
  EXPECT_EQ(int4Result.neonKernelFlavor, "scalar-bridge");
}

TEST(AdapterGenerationContractTest,
     RepeatedGenerateReusesPromptKvCacheWithinSharedRuntimeContext) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("qwen-0.5b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path manifest = RepoRoot() / "tests" / "fixtures" /
                                         "models" / "qwen-0.5b" /
                                         "model.us4manifest";
  ASSERT_TRUE(us4::LoadModelAsset(manifest, asset, &error)) << error;

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult first = adapter->Generate(
      {.prompt = "cache me", .maxTokens = 3, .asset = &asset}, context);
  const us4::GenerationResult second = adapter->Generate(
      {.prompt = "cache me", .maxTokens = 3, .asset = &asset}, context);

  EXPECT_FALSE(first.kvCacheHit);
  EXPECT_TRUE(second.kvCacheHit);
  EXPECT_EQ(first.text, second.text);
  EXPECT_EQ(second.kvPageCount, 1U);
  EXPECT_EQ(second.kvHotPages, 1U);
  EXPECT_EQ(second.prefixCacheEntries, 1U);
}
