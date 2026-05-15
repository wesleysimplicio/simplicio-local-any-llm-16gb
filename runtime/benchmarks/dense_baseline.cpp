#include <chrono>
#include <iostream>

#include "adapters/adapter_registry.h"
#include "core/hardware_probe.h"
#include "core/runtime_context.h"
#include "core/runtime_mode.h"

int main() {
  const us4::HardwareProbeResult probe = us4::HardwareProbe::Detect();
  us4::RuntimeContext context(probe);
  const us4::IUS4V6Adapter* adapter = us4::FindAdapterByModel("qwen-0.5b");
  if (adapter == nullptr) {
    std::cerr << "missing qwen adapter\n";
    return 1;
  }

  adapter->ConfigureRuntime(context);
  const auto start = std::chrono::steady_clock::now();
  const us4::GenerationResult result = adapter->Generate({.prompt = "hi", .maxTokens = 8}, context);
  const auto end = std::chrono::steady_clock::now();
  const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "benchmark=dense_baseline\n";
  std::cout << "recommended_mode=" << us4::ToString(probe.recommendedMode) << "\n";
  std::cout << "model=" << result.modelName << "\n";
  std::cout << "backend=" << result.backend << "\n";
  std::cout << "generated_tokens=" << result.generatedTokens.size() << "\n";
  std::cout << "elapsed_ms=" << elapsedMs << "\n";
  std::cout << "text=" << result.text << "\n";
  return 0;
}
