#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>

#include "adapters/adapter_registry.h"
#include "benchmarks/neon_lowbit_observability.h"
#include "core/hardware_probe.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"
#include "core/runtime_mode.h"

namespace {

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

std::optional<us4::ModelAsset>
LoadOptionalAsset(const std::optional<std::filesystem::path> &manifest) {
  if (!manifest.has_value()) {
    return std::nullopt;
  }

  us4::ModelAsset loaded;
  std::string error;
  if (!us4::LoadModelAsset(*manifest, loaded, &error)) {
    std::cerr << "failed to load manifest " << manifest->string() << ": "
              << error << "\n";
    return std::nullopt;
  }

  return loaded;
}

std::string DisplayPath(const std::string_view rawPath) {
  if (rawPath.empty()) {
    return "";
  }

  std::error_code error;
  const std::filesystem::path rawFilesystemPath =
      std::filesystem::path(std::string(rawPath)).make_preferred();
  const std::filesystem::path canonicalRoot =
      std::filesystem::weakly_canonical(RepoRoot(), error);
  error.clear();
  const std::filesystem::path canonicalAsset =
      std::filesystem::weakly_canonical(rawFilesystemPath, error);
  if (!error) {
    const std::filesystem::path relativePath =
        canonicalAsset.lexically_relative(canonicalRoot);
    if (!relativePath.empty() && relativePath.native()[0] != '.') {
      return relativePath.generic_string();
    }
  }

  return rawFilesystemPath.generic_string();
}

double ComputeMoeHitRate(const us4::GenerationResult &result) {
  const std::size_t denominator = result.moePagerLoads + result.moePagerReuses;
  if (denominator == 0U) {
    return 0.0;
  }
  return static_cast<double>(result.moePagerReuses) /
         static_cast<double>(denominator);
}

double ComputeMoeEvictionRate(const us4::GenerationResult &result) {
  const std::size_t denominator = result.moePagerLoads + result.moePagerReuses;
  if (denominator == 0U) {
    return 0.0;
  }
  return static_cast<double>(result.moePagerEvictions) /
         static_cast<double>(denominator);
}

std::optional<us4::benchmarks::CaseObservation>
RunCase(const us4::HardwareProbeResult &probe, const std::string_view label,
        const std::string_view model,
        const std::optional<us4::ModelAsset> &asset,
        const std::optional<us4::BackendType> requestedBackend) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel(model);
  if (adapter == nullptr) {
    std::cerr << "missing adapter for " << model << "\n";
    return std::nullopt;
  }

  us4::RuntimeContext context(probe);
  adapter->ConfigureRuntime(context);
  const auto start = std::chrono::steady_clock::now();
  const us4::GenerationResult result =
      adapter->Generate({.prompt = "hi",
                         .maxTokens = 8,
                         .asset = asset.has_value() ? &*asset : nullptr,
                         .requestedBackend = requestedBackend},
                        context);
  const auto end = std::chrono::steady_clock::now();
  const auto elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  const us4::benchmarks::CaseObservation observation =
      us4::benchmarks::ObserveCase(label, result, elapsedMs, requestedBackend);
  std::cout << "case=" << observation.label << "\n";
  std::cout << "model=" << result.modelName << "\n";
  std::cout << "requested_backend=" << observation.requestedBackend << "\n";
  std::cout << "observed_backend=" << observation.observedBackend << "\n";
  std::cout << "backend_reason=" << observation.backendReason << "\n";
  std::cout << "asset_format=" << observation.assetFormat << "\n";
  std::cout << "asset_path=" << DisplayPath(observation.assetPath) << "\n";
  std::cout << "mode=" << observation.mode << "\n";
  std::cout << "weight_dtype=" << observation.weightDType << "\n";
  std::cout << "neon_kernel_flavor=" << observation.neonKernelFlavor << "\n";
  std::cout << "dequant_path=" << observation.dequantPath << "\n";
  if (result.moeSelectedExperts > 0 || result.moeShardCount > 0) {
    std::cout << "moe_hit_rate=" << ComputeMoeHitRate(result) << "\n";
    std::cout << "moe_eviction_rate=" << ComputeMoeEvictionRate(result) << "\n";
    std::cout << "moe_router_entropy=" << result.moeRouterEntropy << "\n";
    std::cout << "moe_prefetch_hit_rate=" << result.moePrefetchHitRatio << "\n";
    std::cout << "moe_sparsity_cache_hit_rate="
              << result.moeSparsityCacheHitRatio << "\n";
  }
  std::cout << "generated_tokens=" << observation.generatedTokenCount << "\n";
  std::cout << "elapsed_ms=" << observation.elapsedMs << "\n";
  std::cout << "fell_back=" << (observation.fellBack ? "true" : "false")
            << "\n";
  std::cout << "text_fingerprint=" << observation.textFingerprint << "\n";
  std::cout << "text=" << observation.text << "\n";
  std::cout << "--\n";
  return observation;
}

void PrintRegression(const std::string_view caseLabel,
                     const us4::benchmarks::LowBitRegression &regression) {
  std::cout << "compare_case=" << caseLabel << "\n";
  std::cout << "comparable=" << (regression.comparable ? "true" : "false")
            << "\n";
  std::cout << "text_match=" << (regression.textMatch ? "true" : "false")
            << "\n";
  std::cout << "token_count_match="
            << (regression.tokenCountMatch ? "true" : "false") << "\n";
  std::cout << "dequant_path_match="
            << (regression.dequantPathMatch ? "true" : "false") << "\n";
  std::cout << "neon_kernel_visible="
            << (regression.neonKernelVisible ? "true" : "false") << "\n";
  std::cout << "neon_executed=" << (regression.neonExecuted ? "true" : "false")
            << "\n";
  std::cout << "fallback_observed="
            << (regression.fallbackObserved ? "true" : "false") << "\n";
  std::cout << "speedup_vs_scalar=" << regression.speedupVsScalar << "\n";
  std::cout << "regression_status=" << regression.status << "\n";
  std::cout << "regression_reason=" << regression.reason << "\n";
  std::cout << "--\n";
}

void PrintCorrectnessPlaceholder(const std::string_view caseLabel) {
  std::cout << "compare_case=" << caseLabel << "\n";
  std::cout << "correctness_window_tokens=64\n";
  std::cout << "correctness_delta=pending_external_reference\n";
  std::cout << "correctness_status=placeholder\n";
  std::cout << "correctness_reason=hf-reference-path-not-wired\n";
  std::cout << "--\n";
}

} // namespace

int main() {
  const us4::HardwareProbeResult probe = us4::HardwareProbe::Detect();
  std::cout << "benchmark=dense_baseline\n";
  std::cout << "platform=" << probe.platform << "\n";
  std::cout << "architecture=" << probe.architecture << "\n";
  std::cout << "chip=" << probe.chip << "\n";
  std::cout << "recommended_mode=" << us4::ToString(probe.recommendedMode)
            << "\n";
  std::cout << "neon_vector_bits=" << probe.neonVectorBits << "\n";
  std::cout << "has_performance_cores="
            << (probe.hasPerformanceCores ? "true" : "false") << "\n";
  std::cout << "has_efficiency_cores="
            << (probe.hasEfficiencyCores ? "true" : "false") << "\n";
  std::cout << "--\n";

  const std::filesystem::path repoRoot = RepoRoot();
  if (!RunCase(probe, "dense-default/auto", "qwen-0.5b", std::nullopt,
               std::nullopt)
           .has_value()) {
    return 1;
  }

  // Sprint 04 T04.6: explicit NEON re-bench rows for Qwen and Gemma so the
  // evidence file always exposes scalar vs neon for the same dense adapters.
  if (!RunCase(probe, "dense-qwen/scalar-requested", "qwen-0.5b", std::nullopt,
               us4::BackendType::kScalar)
           .has_value()) {
    return 1;
  }
  if (!RunCase(probe, "dense-qwen/neon-requested", "qwen-0.5b", std::nullopt,
               us4::BackendType::kNeon)
           .has_value()) {
    return 1;
  }
  if (!RunCase(probe, "dense-gemma/scalar-requested", "gemma-2b-it", std::nullopt,
               us4::BackendType::kScalar)
           .has_value()) {
    return 1;
  }
  if (!RunCase(probe, "dense-gemma/neon-requested", "gemma-2b-it", std::nullopt,
               us4::BackendType::kNeon)
           .has_value()) {
    return 1;
  }

  const std::optional<us4::ModelAsset> llamaFixtureAsset =
      LoadOptionalAsset(repoRoot / "tests" / "fixtures" / "models" /
                        "llama-3.1-8b" / "model.us4manifest");
  if (!llamaFixtureAsset.has_value()) {
    return 1;
  }

  if (!RunCase(probe, "llama-fixture/metal-requested", "llama-3.1-8b",
               llamaFixtureAsset, us4::BackendType::kMetal)
           .has_value()) {
    return 1;
  }
  PrintCorrectnessPlaceholder("llama-fixture/metal-requested");

  const std::optional<us4::ModelAsset> llamaGgufAsset =
      LoadOptionalAsset(repoRoot / "tests" / "fixtures" / "models" /
                        "llama-3.1-8b" / "toy-llama.gguf");
  if (!llamaGgufAsset.has_value()) {
    return 1;
  }

  if (!RunCase(probe, "llama-gguf/neon-requested", "llama-3.1-8b",
               llamaGgufAsset, us4::BackendType::kNeon)
           .has_value()) {
    return 1;
  }
  PrintCorrectnessPlaceholder("llama-gguf/neon-requested");

  const std::optional<us4::ModelAsset> int8Asset =
      LoadOptionalAsset(repoRoot / "tests" / "fixtures" / "models" /
                        "bitnet-b1.58-2b" / "model.us4manifest");
  if (!int8Asset.has_value()) {
    return 1;
  }

  const auto int8Scalar =
      RunCase(probe, "lowbit-int8/scalar", "bitnet-b1.58-2b", int8Asset,
              us4::BackendType::kScalarCpu);
  const auto int8Neon = RunCase(probe, "lowbit-int8/neon", "bitnet-b1.58-2b",
                                int8Asset, us4::BackendType::kNeon);
  if (!int8Scalar.has_value() || !int8Neon.has_value()) {
    return 1;
  }
  PrintRegression("lowbit-int8", us4::benchmarks::CompareLowBitObservations(
                                     *int8Scalar, *int8Neon));

  const std::optional<us4::ModelAsset> int4Asset =
      LoadOptionalAsset(repoRoot / "tests" / "fixtures" / "models" /
                        "pt-bitnet-ternary-2b" / "model.us4manifest");
  if (!int4Asset.has_value()) {
    return 1;
  }

  const auto int4Scalar =
      RunCase(probe, "lowbit-int4/scalar", "pt-bitnet-ternary-2b", int4Asset,
              us4::BackendType::kScalarCpu);
  const auto int4Neon =
      RunCase(probe, "lowbit-int4/neon", "pt-bitnet-ternary-2b", int4Asset,
              us4::BackendType::kNeon);
  if (!int4Scalar.has_value() || !int4Neon.has_value()) {
    return 1;
  }
  PrintRegression("lowbit-int4", us4::benchmarks::CompareLowBitObservations(
                                     *int4Scalar, *int4Neon));

  return 0;
}
