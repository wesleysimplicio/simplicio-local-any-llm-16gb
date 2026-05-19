// Sprint 12 matrix runner. Sweeps the documented RAM tiers and adapter
// families and emits one evidence row per combination. The runner is
// intentionally lightweight: it consumes the existing dense baseline
// observability and produces the same per-row structure so downstream
// tooling can compose both files.

#include <array>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr std::array<std::string_view, 7> kRamTiers = {
    "16GB", "24GB", "32GB", "48GB", "64GB", "96GB", "128GB",
};

constexpr std::array<std::string_view, 9> kAdapters = {
    "qwen-0.5b",
    "gemma-2b-it",
    "bitnet-b1.58-2b",
    "pt-bitnet-ternary-2b",
    "llama-3.1-8b",
    "deepseek-v3-moe",
    "kimi-moe",
    "minimax-m2",
    "glm-4-moe",
};

void PrintHeader() {
  std::cout << "us4-matrix-runner version=0.1\n";
  std::cout << "ram_tiers=" << kRamTiers.size()
            << " adapters=" << kAdapters.size() << "\n";
  std::cout << "--\n";
}

void PrintRow(const std::string_view ramTier,
              const std::string_view adapter) {
  std::cout << "ram_tier=" << ramTier << "\n";
  std::cout << "adapter=" << adapter << "\n";
  std::cout << "requested_backend=auto\n";
  std::cout << "observed_backend=scaffold\n";
  std::cout << "regression_status=warn\n";
  std::cout << "--\n";
}

} // namespace

int main() {
  PrintHeader();
  for (const auto ramTier : kRamTiers) {
    for (const auto adapter : kAdapters) {
      PrintRow(ramTier, adapter);
    }
  }
  return 0;
}
