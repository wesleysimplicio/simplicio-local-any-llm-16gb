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
  const us4::IUS4V6Adapter *kimi = us4::FindAdapterByModel("kimi");
  const us4::IUS4V6Adapter *minimax = us4::FindAdapterByModel("minimax");

  ASSERT_NE(qwen, nullptr);
  ASSERT_NE(gemma, nullptr);
  ASSERT_NE(deepseek, nullptr);
  ASSERT_NE(kimi, nullptr);
  ASSERT_NE(minimax, nullptr);
  EXPECT_EQ(qwen->Architecture(), us4::ArchitectureType::kDense);
  EXPECT_EQ(gemma->Architecture(), us4::ArchitectureType::kDense);
  EXPECT_EQ(deepseek->Architecture(), us4::ArchitectureType::kMoe);
  EXPECT_EQ(kimi->Architecture(), us4::ArchitectureType::kMoe);
  EXPECT_EQ(minimax->Architecture(), us4::ArchitectureType::kMoe);
}

TEST(AdapterGenerationContractTest,
     DeepSeekMoeAdapterConsumesRouteMetadataAndReusesPagerWithinContext) {
  const us4::IUS4V6Adapter *adapter =
      us4::FindAdapterByModel("deepseek-v2-lite");
  ASSERT_NE(adapter, nullptr);

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult first = adapter->Generate(
      {.prompt = "code logic runtime", .maxTokens = 3}, context);
  const us4::GenerationResult second = adapter->Generate(
      {.prompt = "code logic runtime", .maxTokens = 3}, context);
  const us4::GenerationResult third =
      adapter->Generate({.prompt = "wide context", .maxTokens = 3}, context);

  EXPECT_EQ(first.family, "deepseek");
  EXPECT_EQ(first.moeSelectedExperts, 2U);
  EXPECT_GT(first.moeRouterEntropy, 0.0F);
  EXPECT_GT(first.moeSelectedMass, 0.0F);
  EXPECT_NE(first.text.find("moe-route"), std::string::npos);
  EXPECT_NE(first.text.find("e0"), std::string::npos);
  EXPECT_NE(first.text.find("e1"), std::string::npos);
  EXPECT_EQ(first.moePagerLoads, 2U);
  EXPECT_EQ(first.moePagerReuses, 0U);
  EXPECT_EQ(first.moePagerEvictions, 0U);
  EXPECT_EQ(first.moeResidentExperts, 2U);

  EXPECT_EQ(second.moePagerLoads, 2U);
  EXPECT_GE(second.moePagerReuses, 2U);
  EXPECT_EQ(second.moePagerEvictions, 0U);
  EXPECT_NE(second.text.find("moe-route"), std::string::npos);

  EXPECT_EQ(third.moeSelectedExperts, 2U);
  EXPECT_GE(third.moePagerLoads, 3U);
  EXPECT_GE(third.moePagerEvictions, 1U);
  EXPECT_EQ(third.moeResidentExperts, 2U);
  EXPECT_NE(third.text.find("moe-route"), std::string::npos);
}

TEST(AdapterGenerationContractTest,
     KimiMoeAdapterConsumesRouteMetadataAndReusesPagerWithinContext) {
  const us4::IUS4V6Adapter *adapter =
      us4::FindAdapterByModel("kimi-k2-instruct");
  ASSERT_NE(adapter, nullptr);

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult first =
      adapter->Generate({.prompt = "smart context", .maxTokens = 3}, context);
  const us4::GenerationResult second =
      adapter->Generate({.prompt = "smart context", .maxTokens = 3}, context);
  const us4::GenerationResult third =
      adapter->Generate({.prompt = "fast local", .maxTokens = 3}, context);

  EXPECT_EQ(first.family, "kimi");
  EXPECT_EQ(first.moeSelectedExperts, 2U);
  EXPECT_GT(first.moeRouterEntropy, 0.0F);
  EXPECT_GT(first.moeSelectedMass, 0.0F);
  EXPECT_NE(first.text.find("kimi-route"), std::string::npos);
  EXPECT_NE(first.text.find("e1"), std::string::npos);
  EXPECT_NE(first.text.find("e3"), std::string::npos);
  EXPECT_EQ(first.moePagerLoads, 2U);
  EXPECT_EQ(first.moePagerReuses, 0U);
  EXPECT_EQ(first.moePagerEvictions, 0U);
  EXPECT_EQ(first.moeResidentExperts, 2U);

  EXPECT_EQ(second.moePagerLoads, 2U);
  EXPECT_GE(second.moePagerReuses, 2U);
  EXPECT_EQ(second.moePagerEvictions, 0U);
  EXPECT_NE(second.text.find("kimi-route"), std::string::npos);

  EXPECT_EQ(third.moeSelectedExperts, 2U);
  EXPECT_GE(third.moePagerLoads, 3U);
  EXPECT_GE(third.moePagerEvictions, 1U);
  EXPECT_EQ(third.moeResidentExperts, 2U);
  EXPECT_NE(third.text.find("kimi-route"), std::string::npos);
}

TEST(AdapterGenerationContractTest,
     MiniMaxMoeAdapterConsumesRouteMetadataAndReusesPagerWithinContext) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("minimax-m2");
  ASSERT_NE(adapter, nullptr);

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult first = adapter->Generate(
      {.prompt = "image audio fusion", .maxTokens = 3}, context);
  const us4::GenerationResult second = adapter->Generate(
      {.prompt = "image audio fusion", .maxTokens = 3}, context);
  const us4::GenerationResult third = adapter->Generate(
      {.prompt = "logic wide context", .maxTokens = 3}, context);

  EXPECT_EQ(first.family, "minimax");
  EXPECT_EQ(first.moeSelectedExperts, 2U);
  EXPECT_GT(first.moeRouterEntropy, 0.0F);
  EXPECT_GT(first.moeSelectedMass, 0.0F);
  EXPECT_NE(first.text.find("minimax-route"), std::string::npos);
  EXPECT_EQ(first.moePagerLoads, 2U);
  EXPECT_EQ(first.moePagerReuses, 0U);
  EXPECT_EQ(first.moePagerEvictions, 0U);
  EXPECT_EQ(first.moeResidentExperts, 2U);

  EXPECT_EQ(second.moePagerLoads, 2U);
  EXPECT_GE(second.moePagerReuses, 2U);
  EXPECT_EQ(second.moePagerEvictions, 0U);
  EXPECT_NE(second.text.find("minimax-route"), std::string::npos);

  EXPECT_EQ(third.moeSelectedExperts, 2U);
  EXPECT_GE(third.moePagerLoads, 3U);
  EXPECT_GE(third.moePagerEvictions, 1U);
  EXPECT_EQ(third.moeResidentExperts, 2U);
  EXPECT_NE(third.text.find("minimax-route"), std::string::npos);
}

TEST(AdapterGenerationContractTest,
     MoeAdaptersSurfaceShardAwareLoaderTelemetryWhenAssetIsProvided) {
  const std::array<std::pair<const char *, std::filesystem::path>, 3> kCases = {
      {
          {"deepseek-v2-lite", RepoRoot() / "tests" / "fixtures" / "models" /
                                   "deepseek-v2-lite" / "model.us4manifest"},
          {"kimi-k2-instruct", RepoRoot() / "tests" / "fixtures" / "models" /
                                   "kimi-k2-instruct" / "model.us4manifest"},
          {"minimax-m2", RepoRoot() / "tests" / "fixtures" / "models" /
                             "minimax-m2" / "model.us4manifest"},
      }};

  for (const auto &[modelName, manifestPath] : kCases) {
    SCOPED_TRACE(modelName);

    const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel(modelName);
    ASSERT_NE(adapter, nullptr);

    us4::ModelAsset asset;
    std::string error;
    ASSERT_TRUE(us4::LoadModelAsset(manifestPath, asset, &error)) << error;

    us4::RuntimeContext context(MakeProbe());
    adapter->ConfigureRuntime(context);
    const us4::GenerationResult result = adapter->Generate(
        {.prompt = "smart context", .maxTokens = 3, .asset = &asset}, context);

    EXPECT_EQ(result.moeShardCount, 2U);
    EXPECT_EQ(result.moeActiveExperts, 2U);
    EXPECT_TRUE(result.moeLazyLoad);
    EXPECT_EQ(result.assetFormat, "fixture-manifest");
  }
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

TEST(AdapterGenerationContractTest,
     LlamaDirectoryManifestKeepsFixturePromptAndNeonContractVisible) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("llama-3.1-8b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path manifestDirectory =
      RepoRoot() / "tests" / "fixtures" / "models" / "llama-3.1-8b";
  ASSERT_TRUE(us4::LoadModelAsset(manifestDirectory, asset, &error)) << error;

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result = adapter->Generate(
      {.prompt = "", .maxTokens = 3, .asset = &asset}, context);

  ASSERT_EQ(result.promptTokens.size(), 1U);
  EXPECT_EQ(result.promptTokens[0], "hello");
  EXPECT_EQ(result.family, "llama");
  EXPECT_EQ(result.modelName, "llama-3.1-8b-fixture");
  EXPECT_EQ(result.assetFormat, "fixture-manifest");
  EXPECT_EQ(std::filesystem::path(result.assetPath).filename(),
            "model.us4manifest");
  EXPECT_EQ(result.backend, "neon");
  EXPECT_EQ(result.backendReason, "auto-neon");
  EXPECT_FALSE(result.fellBack);
  EXPECT_EQ(result.weightDType, "fp16");
  EXPECT_EQ(result.neonKernelFlavor, "fp16-lane8");
  EXPECT_EQ(result.dequantPath, "none");
  EXPECT_FALSE(result.kvCacheHit);
  EXPECT_EQ(result.kvPageCount, 1U);
  EXPECT_EQ(result.kvHotPages, 1U);
  EXPECT_EQ(result.generatedTokens.size(), 3U);
  EXPECT_EQ(result.text.find("gqa-error"), std::string::npos);
}

TEST(AdapterGenerationContractTest,
     LlamaGgufAssetSurfacesLoaderAndFallbackTelemetry) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("llama-3.1-8b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path ggufPath = RepoRoot() / "tests" / "fixtures" /
                                         "models" / "llama-3.1-8b" /
                                         "toy-llama.gguf";
  ASSERT_TRUE(us4::LoadModelAsset(ggufPath, asset, &error)) << error;

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result =
      adapter->Generate({.prompt = "hello",
                         .maxTokens = 4,
                         .asset = &asset,
                         .requestedBackend = us4::BackendType::kMetal},
                        context);

  EXPECT_EQ(result.family, "llama");
  EXPECT_EQ(result.modelName, "toy-llama");
  EXPECT_EQ(result.assetFormat, "gguf");
  EXPECT_EQ(result.assetPath, ggufPath.string());
  EXPECT_EQ(result.backend, "neon");
  EXPECT_EQ(result.backendReason, "requested-backend-unavailable");
  EXPECT_TRUE(result.fellBack);
  EXPECT_EQ(result.sharedAllocations, 0U);
  EXPECT_EQ(result.metalDispatches, 0U);
  EXPECT_FALSE(result.mlxPlanBuilt);
  EXPECT_EQ(result.weightDType, "fp16");
  EXPECT_EQ(result.neonKernelFlavor, "fp16-lane8");
  EXPECT_EQ(result.dequantPath, "none");
  ASSERT_EQ(result.promptTokens.size(), 1U);
  EXPECT_EQ(result.promptTokens[0], "hello");
  EXPECT_EQ(result.generatedTokens.size(), 4U);
  EXPECT_EQ(result.kvPageCount, 1U);
}

TEST(AdapterGenerationContractTest,
     RepeatedLlamaGenerateReusesPromptKvWithinSharedRuntimeContext) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("llama-3.1-8b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path manifestDirectory =
      RepoRoot() / "tests" / "fixtures" / "models" / "llama-3.1-8b";
  ASSERT_TRUE(us4::LoadModelAsset(manifestDirectory, asset, &error)) << error;

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult first = adapter->Generate(
      {.prompt = "hello llama", .maxTokens = 3, .asset = &asset}, context);
  const us4::GenerationResult second = adapter->Generate(
      {.prompt = "hello llama", .maxTokens = 3, .asset = &asset}, context);

  EXPECT_FALSE(first.kvCacheHit);
  EXPECT_TRUE(second.kvCacheHit);
  EXPECT_EQ(first.text, second.text);
  EXPECT_EQ(first.text.find("gqa-error"), std::string::npos);
  EXPECT_EQ(second.backend, "neon");
  EXPECT_EQ(second.kvPageCount, 1U);
  EXPECT_EQ(second.kvHotPages, 1U);
  EXPECT_EQ(second.prefixCacheEntries, 1U);
}

TEST(AdapterGenerationContractTest,
     LlamaDedicatedKvCacheStaysPartitionedByAssetSeedWithinSharedContext) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("llama-3.1-8b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset baselineAsset;
  std::string error;
  const std::filesystem::path manifestDirectory =
      RepoRoot() / "tests" / "fixtures" / "models" / "llama-3.1-8b";
  ASSERT_TRUE(us4::LoadModelAsset(manifestDirectory, baselineAsset, &error))
      << error;

  us4::ModelAsset alternateSeedAsset = baselineAsset;
  alternateSeedAsset.seed += 7U;

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult baselineFirst = adapter->Generate(
      {.prompt = "hello llama", .maxTokens = 3, .asset = &baselineAsset},
      context);
  const us4::GenerationResult alternateSeed = adapter->Generate(
      {.prompt = "hello llama", .maxTokens = 3, .asset = &alternateSeedAsset},
      context);
  const us4::GenerationResult alternateSeedRepeat = adapter->Generate(
      {.prompt = "hello llama", .maxTokens = 3, .asset = &alternateSeedAsset},
      context);

  EXPECT_FALSE(baselineFirst.kvCacheHit);
  EXPECT_FALSE(alternateSeed.kvCacheHit);
  EXPECT_TRUE(alternateSeedRepeat.kvCacheHit);
  EXPECT_EQ(alternateSeed.backend, "neon");
  EXPECT_EQ(alternateSeed.kvPageCount, 2U);
  EXPECT_EQ(alternateSeedRepeat.kvPageCount, 2U);
  EXPECT_EQ(alternateSeedRepeat.kvHotPages, 1U);
  EXPECT_EQ(alternateSeedRepeat.kvWarmPages, 1U);
  EXPECT_EQ(alternateSeedRepeat.prefixCacheEntries, 2U);
}

TEST(AdapterGenerationContractTest,
     LlamaScalarBackendKeepsFp16AssetsRunnableWithoutMatmulError) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("llama-3.1-8b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path manifestDirectory =
      RepoRoot() / "tests" / "fixtures" / "models" / "llama-3.1-8b";
  ASSERT_TRUE(us4::LoadModelAsset(manifestDirectory, asset, &error)) << error;

  us4::RuntimeContext context(MakeProbe());
  adapter->ConfigureRuntime(context);

  const us4::GenerationResult result =
      adapter->Generate({.prompt = "hello",
                         .maxTokens = 3,
                         .asset = &asset,
                         .requestedBackend = us4::BackendType::kScalarCpu},
                        context);

  EXPECT_EQ(result.backend, "scalar");
  EXPECT_EQ(result.backendReason, "requested");
  EXPECT_FALSE(result.fellBack);
  EXPECT_EQ(result.weightDType, "fp16");
  EXPECT_EQ(result.neonKernelFlavor, "none");
  EXPECT_EQ(result.sharedAllocations, 0U);
  EXPECT_EQ(result.metalDispatches, 0U);
  EXPECT_EQ(result.generatedTokens.size(), 3U);
  EXPECT_EQ(result.generatedTokens[0], "llama");
  EXPECT_EQ(result.text.find("matmul-error"), std::string::npos);
  EXPECT_EQ(result.kvPageCount, 1U);
}

TEST(AdapterGenerationContractTest,
     LlamaDedicatedPathKeepsKvReuseScopedToSharedRuntimeContext) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("llama-3.1-8b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path manifestDirectory =
      RepoRoot() / "tests" / "fixtures" / "models" / "llama-3.1-8b";
  ASSERT_TRUE(us4::LoadModelAsset(manifestDirectory, asset, &error)) << error;

  us4::RuntimeContext firstContext(MakeProbe());
  firstContext.SetMode(us4::RuntimeMode::kMicro);
  const us4::GenerationResult first = adapter->Generate(
      {.prompt = "hello llama", .maxTokens = 3, .asset = &asset}, firstContext);

  us4::RuntimeContext secondContext(MakeProbe());
  secondContext.SetMode(us4::RuntimeMode::kMicro);
  const us4::GenerationResult second = adapter->Generate(
      {.prompt = "hello llama", .maxTokens = 3, .asset = &asset},
      secondContext);

  EXPECT_FALSE(first.kvCacheHit);
  EXPECT_FALSE(first.kvRestoredFromColdStore);
  EXPECT_EQ(first.kvSummaryRows, 0U);
  EXPECT_FALSE(second.kvCacheHit);
  EXPECT_FALSE(second.kvRestoredFromColdStore);
  EXPECT_EQ(second.kvSummaryRows, 0U);
  EXPECT_EQ(second.kvPageCount, 1U);
  EXPECT_EQ(second.kvHotPages, 1U);
}

TEST(AdapterGenerationContractTest,
     LlamaMicroModeCanRestorePromptKvFromColdStoreWithSummaryTelemetry) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("llama-3.1-8b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path manifestDirectory =
      RepoRoot() / "tests" / "fixtures" / "models" / "llama-3.1-8b";
  ASSERT_TRUE(us4::LoadModelAsset(manifestDirectory, asset, &error)) << error;

  std::filesystem::remove_all(RepoRoot() / "build" / "kv-cold-store");

  us4::RuntimeContext firstContext(MakeProbe());
  firstContext.SetMode(us4::RuntimeMode::kMicro);
  const us4::GenerationResult first =
      adapter->Generate({.prompt = "one two three four five six seven",
                         .maxTokens = 2,
                         .asset = &asset},
                        firstContext);

  EXPECT_GT(first.kvSummaryRows, 0U);

  us4::RuntimeContext secondContext(MakeProbe());
  secondContext.SetMode(us4::RuntimeMode::kMicro);
  const us4::GenerationResult second =
      adapter->Generate({.prompt = "one two three four five six seven",
                         .maxTokens = 2,
                         .asset = &asset},
                        secondContext);

  EXPECT_TRUE(second.kvCacheHit);
  EXPECT_TRUE(second.kvRestoredFromColdStore);
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

TEST(AdapterGenerationContractTest,
     MicroModeCanRestorePromptKvFromColdStoreWithSummaryTelemetry) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel("qwen-0.5b");
  ASSERT_NE(adapter, nullptr);

  us4::ModelAsset asset;
  std::string error;
  const std::filesystem::path manifest = RepoRoot() / "tests" / "fixtures" /
                                         "models" / "qwen-0.5b" /
                                         "model.us4manifest";
  ASSERT_TRUE(us4::LoadModelAsset(manifest, asset, &error)) << error;

  std::filesystem::remove_all(RepoRoot() / "build" / "kv-cold-store");

  us4::RuntimeContext firstContext(MakeProbe());
  firstContext.SetMode(us4::RuntimeMode::kMicro);
  const us4::GenerationResult first =
      adapter->Generate({.prompt = "one two three four five six seven",
                         .maxTokens = 2,
                         .asset = &asset},
                        firstContext);

  EXPECT_GT(first.kvSummaryRows, 0U);

  us4::RuntimeContext secondContext(MakeProbe());
  secondContext.SetMode(us4::RuntimeMode::kMicro);
  const us4::GenerationResult second =
      adapter->Generate({.prompt = "one two three four five six seven",
                         .maxTokens = 2,
                         .asset = &asset},
                        secondContext);

  EXPECT_TRUE(second.kvCacheHit);
  EXPECT_TRUE(second.kvRestoredFromColdStore);
}
