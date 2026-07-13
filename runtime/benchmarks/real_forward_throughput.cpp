// Issue #81.12: none of #82-#91 measured performance -- only functional
// correctness (the output is the expected token). This benchmark measures
// tokens/s, decode latency, and process memory over the REAL forward path
// (real embedding/lm_head weights, see #85/#105) and reports it side by
// side with the same adapter's fully synthetic path, so the comparison the
// epic's DoD asks for is explicit instead of implied.
//
// Scope note: this runs on CPU (scalar/NEON-selectable) only. Metal/MLX/ANE
// real execution is blocked by #86 (no macOS/Apple Silicon hardware in this
// environment) and is out of scope here.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "adapters/adapter_registry.h"
#include "core/hardware_probe.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"

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

// Resident set size in KiB, or -1 when unavailable (non-Linux). This is a
// coarse, whole-process signal -- good enough to flag a gross regression
// or leak across repeated runs, not a precise per-allocation profile.
long ReadResidentSetSizeKiB() {
#if defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream stream(line.substr(6));
      long kib = -1;
      stream >> kib;
      return kib;
    }
  }
  return -1;
#else
  return -1;
#endif
}

struct RunStats {
  std::vector<double> tokensPerSecond;
  std::vector<double> latencyMs;
  long rssBeforeKiB = -1;
  long rssAfterKiB = -1;
};

double Mean(const std::vector<double> &values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

RunStats RunRepeated(const us4::IUS4V6Adapter &adapter,
                     const us4::HardwareProbeResult &probe,
                     const us4::ModelAsset *asset,
                     const us4::BackendType backend,
                     const std::size_t maxTokens, const std::size_t repeats) {
  RunStats stats;
  stats.rssBeforeKiB = ReadResidentSetSizeKiB();

  for (std::size_t iteration = 0; iteration < repeats; ++iteration) {
    us4::RuntimeContext context(probe);
    adapter.ConfigureRuntime(context);

    const auto start = std::chrono::steady_clock::now();
    const us4::GenerationResult result =
        adapter.Generate({.prompt = "",
                          .maxTokens = maxTokens,
                          .asset = asset,
                          .requestedBackend = backend},
                         context);
    const auto end = std::chrono::steady_clock::now();

    const double elapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    const double tokens = static_cast<double>(result.generatedTokens.size());
    stats.latencyMs.push_back(elapsedMs);
    stats.tokensPerSecond.push_back(
        elapsedMs > 0.0 ? tokens / (elapsedMs / 1000.0) : 0.0);
  }

  stats.rssAfterKiB = ReadResidentSetSizeKiB();
  return stats;
}

void PrintCase(const std::string_view label, const RunStats &real,
               const RunStats &synthetic) {
  std::cout << "case=" << label << "\n";
  std::cout << "real_tokens_per_second_mean=" << Mean(real.tokensPerSecond)
            << "\n";
  std::cout << "real_latency_ms_mean=" << Mean(real.latencyMs) << "\n";
  std::cout << "real_latency_ms_min="
            << *std::min_element(real.latencyMs.begin(), real.latencyMs.end())
            << "\n";
  std::cout << "real_latency_ms_max="
            << *std::max_element(real.latencyMs.begin(), real.latencyMs.end())
            << "\n";
  std::cout << "synthetic_tokens_per_second_mean="
            << Mean(synthetic.tokensPerSecond) << "\n";
  std::cout << "synthetic_latency_ms_mean=" << Mean(synthetic.latencyMs)
            << "\n";
  const double meanReal = Mean(real.latencyMs);
  const double meanSynthetic = Mean(synthetic.latencyMs);
  std::cout << "real_vs_synthetic_latency_ratio="
            << (meanSynthetic > 0.0 ? meanReal / meanSynthetic : 0.0) << "\n";
  std::cout << "real_rss_before_kib=" << real.rssBeforeKiB << "\n";
  std::cout << "real_rss_after_kib=" << real.rssAfterKiB << "\n";
  std::cout << "rss_delta_kib="
            << (real.rssBeforeKiB >= 0 && real.rssAfterKiB >= 0
                    ? real.rssAfterKiB - real.rssBeforeKiB
                    : -1)
            << "\n";
  std::cout << "--\n";
}

std::optional<us4::ModelAsset> LoadAsset(const std::filesystem::path &path) {
  us4::ModelAsset asset;
  std::string error;
  if (!us4::LoadModelAsset(path, asset, &error)) {
    std::cerr << "failed to load " << path.string() << ": " << error << "\n";
    return std::nullopt;
  }
  return asset;
}

} // namespace

int main() {
  const us4::HardwareProbeResult probe = us4::HardwareProbe::Detect();
  constexpr std::size_t kMaxTokens = 64;
  constexpr std::size_t kRepeats = 5;

  std::cout << "benchmark=real_forward_throughput\n";
  std::cout << "platform=" << probe.platform << "\n";
  std::cout << "architecture=" << probe.architecture << "\n";
  std::cout << "chip=" << probe.chip << "\n";
  std::cout << "has_neon=" << (probe.hasNeon ? "true" : "false") << "\n";
  std::cout << "max_tokens_per_run=" << kMaxTokens << "\n";
  std::cout << "repeats_per_case=" << kRepeats << "\n";
  std::cout << "decoding=greedy-argmax-deterministic-no-sampling-temperature\n";
  std::cout << "--\n";

  const std::filesystem::path repoRoot = RepoRoot();

  const us4::IUS4V6Adapter *qwen = us4::FindAdapterByModel("qwen-0.5b");
  if (qwen == nullptr) {
    std::cerr << "missing qwen adapter\n";
    return 1;
  }
  const std::optional<us4::ModelAsset> qwenRealAsset =
      LoadAsset(repoRoot / "tests" / "fixtures" / "models" / "toy-dense-real" /
                "toy-dense-real.safetensors");
  if (!qwenRealAsset.has_value()) {
    return 1;
  }
  const RunStats qwenRealScalar =
      RunRepeated(*qwen, probe, &*qwenRealAsset, us4::BackendType::kScalarCpu,
                  kMaxTokens, kRepeats);
  const RunStats qwenSyntheticScalar =
      RunRepeated(*qwen, probe, nullptr, us4::BackendType::kScalarCpu,
                  kMaxTokens, kRepeats);
  PrintCase("qwen-real-vs-synthetic/scalar", qwenRealScalar,
            qwenSyntheticScalar);

  const RunStats qwenRealNeon =
      RunRepeated(*qwen, probe, &*qwenRealAsset, us4::BackendType::kNeon,
                  kMaxTokens, kRepeats);
  const RunStats qwenSyntheticNeon = RunRepeated(
      *qwen, probe, nullptr, us4::BackendType::kNeon, kMaxTokens, kRepeats);
  PrintCase("qwen-real-vs-synthetic/neon-requested", qwenRealNeon,
            qwenSyntheticNeon);

  const us4::IUS4V6Adapter *llama = us4::FindAdapterByModel("llama-3.1-8b");
  if (llama == nullptr) {
    std::cerr << "missing llama adapter\n";
    return 1;
  }
  const std::optional<us4::ModelAsset> llamaRealAsset =
      LoadAsset(repoRoot / "tests" / "fixtures" / "models" / "toy-llama-real" /
                "toy-llama-real.safetensors");
  if (!llamaRealAsset.has_value()) {
    return 1;
  }
  const RunStats llamaRealNeon =
      RunRepeated(*llama, probe, &*llamaRealAsset, us4::BackendType::kNeon,
                  kMaxTokens, kRepeats);
  const RunStats llamaSyntheticNeon = RunRepeated(
      *llama, probe, nullptr, us4::BackendType::kNeon, kMaxTokens, kRepeats);
  PrintCase("llama-real-vs-synthetic/neon-gqa", llamaRealNeon,
            llamaSyntheticNeon);

  return 0;
}
