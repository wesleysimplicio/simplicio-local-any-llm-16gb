#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "adapters/adapter_registry.h"
#include "core/hardware_probe.h"
#include "core/ius4v6_adapter.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"
#include "core/runtime_mode.h"
#include "metal/command_queue.h"
#include "metal/device_info.h"
#include "us4/version.h"

namespace {

std::string_view
ArchitectureToString(const us4::ArchitectureType architecture) {
  switch (architecture) {
  case us4::ArchitectureType::kDense:
    return "dense";
  case us4::ArchitectureType::kMoe:
    return "moe";
  case us4::ArchitectureType::kTernary:
    return "ternary";
  case us4::ArchitectureType::kUnknown:
    return "unknown";
  }
  return "unknown";
}

std::string EscapeJson(const std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(ch);
      break;
    }
  }
  return escaped;
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

double ComputeMoeSparsityCacheHitRate(const us4::GenerationResult &result) {
  const std::size_t denominator =
      result.moeSparsityCacheHits + result.moeSparsityCacheMisses;
  if (denominator == 0U) {
    return 0.0;
  }
  return static_cast<double>(result.moeSparsityCacheHits) /
         static_cast<double>(denominator);
}

double ComputeMultimodalCacheHitRate(const us4::GenerationResult &result) {
  const std::size_t denominator =
      result.multimodalCacheHits + result.multimodalCacheMisses;
  if (denominator == 0U) {
    return 0.0;
  }
  return static_cast<double>(result.multimodalCacheHits) /
         static_cast<double>(denominator);
}

void PrintHelp() {
  std::cout << "US4 V6 Apple Edition CLI\n"
            << "Usage:\n"
            << "  us4-cli --version\n"
            << "  us4-cli --probe [--json]\n"
            << "  us4-cli --mode auto [--json]\n"
            << "  us4-cli list-models [--json]\n"
            << "  us4-cli run --model <name> [--model-path <path>] [--backend "
               "<scalar|neon|mlx|metal|ane>] --prompt <text> [--max-tokens N] "
               "[--json]\n";
}

void PrintProbeText(const us4::HardwareProbeResult &probe) {
  const us4::MetalCommandQueue metalQueue(probe);
  const us4::MetalDeviceInfo &metalDevice = metalQueue.Device();
  const us4::MetalQueueProfile &metalProfile = metalQueue.Profile();
  std::cout << "US4 V6 Apple Edition\n"
            << "version: " << us4::kUs4Version << "\n"
            << "platform: " << probe.platform << "\n"
            << "architecture: " << probe.architecture << "\n"
            << "chip: " << probe.chip << "\n"
            << "memory_gib: " << probe.unifiedMemoryGiB << "\n"
            << "is_apple_silicon: " << (probe.isAppleSilicon ? "true" : "false")
            << "\n"
            << "has_mlx: " << (probe.hasMlx ? "true" : "false") << "\n"
            << "has_metal: " << (probe.hasMetal ? "true" : "false") << "\n"
            << "has_neon: " << (probe.hasNeon ? "true" : "false") << "\n"
            << "has_ane: " << (probe.hasAne ? "true" : "false") << "\n"
            << "neon_vector_bits: " << probe.neonVectorBits << "\n"
            << "has_performance_cores: "
            << (probe.hasPerformanceCores ? "true" : "false") << "\n"
            << "has_efficiency_cores: "
            << (probe.hasEfficiencyCores ? "true" : "false") << "\n"
            << "metal_device: " << metalDevice.deviceName << "\n"
            << "metal_queue_label: " << metalDevice.queueLabel << "\n"
            << "metal_threads_per_group: "
            << metalDevice.maxThreadsPerThreadgroup << "\n"
            << "supports_unified_memory: "
            << (metalDevice.supportsUnifiedMemory ? "true" : "false") << "\n"
            << "metal_init_stage: " << us4::ToString(metalProfile.stage) << "\n"
            << "metal_queue_created: "
            << (metalProfile.queueCreated ? "true" : "false") << "\n"
            << "metal_autorelease_boundary_requested: "
            << (metalProfile.requiresAutoreleaseBoundary ? "true" : "false")
            << "\n"
            << "metal_objc_boundary_supported: "
            << (metalProfile.hostSupportsObjectiveCBoundary ? "true" : "false")
            << "\n"
            << "metal_reason: " << metalQueue.Reason() << "\n"
            << "recommended_mode: " << us4::ToString(probe.recommendedMode)
            << "\n";
}

void PrintProbeJson(const us4::HardwareProbeResult &probe) {
  const us4::MetalCommandQueue metalQueue(probe);
  const us4::MetalDeviceInfo &metalDevice = metalQueue.Device();
  const us4::MetalQueueProfile &metalProfile = metalQueue.Profile();
  std::cout << "{"
            << "\"version\":\"" << EscapeJson(us4::kUs4Version) << "\","
            << "\"platform\":\"" << EscapeJson(probe.platform) << "\","
            << "\"architecture\":\"" << EscapeJson(probe.architecture) << "\","
            << "\"chip\":\"" << EscapeJson(probe.chip) << "\","
            << "\"memory_gib\":" << probe.unifiedMemoryGiB << ","
            << "\"is_apple_silicon\":"
            << (probe.isAppleSilicon ? "true" : "false") << ","
            << "\"has_mlx\":" << (probe.hasMlx ? "true" : "false") << ","
            << "\"has_metal\":" << (probe.hasMetal ? "true" : "false") << ","
            << "\"has_neon\":" << (probe.hasNeon ? "true" : "false") << ","
            << "\"has_ane\":" << (probe.hasAne ? "true" : "false") << ","
            << "\"neon_vector_bits\":" << probe.neonVectorBits << ","
            << "\"has_performance_cores\":"
            << (probe.hasPerformanceCores ? "true" : "false") << ","
            << "\"has_efficiency_cores\":"
            << (probe.hasEfficiencyCores ? "true" : "false") << ","
            << "\"metal_device\":\"" << EscapeJson(metalDevice.deviceName)
            << "\","
            << "\"metal_queue_label\":\"" << EscapeJson(metalDevice.queueLabel)
            << "\","
            << "\"metal_threads_per_group\":"
            << metalDevice.maxThreadsPerThreadgroup << ","
            << "\"supports_unified_memory\":"
            << (metalDevice.supportsUnifiedMemory ? "true" : "false") << ","
            << "\"metal_init_stage\":\""
            << EscapeJson(us4::ToString(metalProfile.stage)) << "\","
            << "\"metal_queue_created\":"
            << (metalProfile.queueCreated ? "true" : "false") << ","
            << "\"metal_autorelease_boundary_requested\":"
            << (metalProfile.requiresAutoreleaseBoundary ? "true" : "false")
            << ","
            << "\"metal_objc_boundary_supported\":"
            << (metalProfile.hostSupportsObjectiveCBoundary ? "true" : "false")
            << ","
            << "\"metal_reason\":\"" << EscapeJson(metalQueue.Reason()) << "\","
            << "\"recommended_mode\":\"" << us4::ToString(probe.recommendedMode)
            << "\""
            << "}\n";
}

void PrintRunText(const us4::GenerationResult &result) {
  std::cout
      << "family: " << result.family << "\n"
      << "model: " << result.modelName << "\n"
      << "asset_format: " << result.assetFormat << "\n"
      << "asset_path: "
      << (result.assetPath.empty() ? "<builtin>" : result.assetPath) << "\n"
      << "draft_model_format: " << result.draftModelFormat << "\n"
      << "draft_model_path: "
      << (result.draftModelPath.empty() ? "<none>" : result.draftModelPath)
      << "\n"
      << "speculative_strategy: " << result.speculativeStrategy << "\n"
      << "speculative_session_scope: " << result.speculativeSessionScope << "\n"
      << "mode: " << us4::ToString(result.mode) << "\n"
      << "backend: " << result.backend << "\n"
      << "backend_reason: " << result.backendReason << "\n"
      << "fallback: " << (result.fellBack ? "true" : "false") << "\n"
      << "shared_allocations: " << result.sharedAllocations << "\n"
      << "metal_dispatches: " << result.metalDispatches << "\n"
      << "mlx_operation_count: " << result.mlxOperationCount << "\n"
      << "kv_cache_hit: " << (result.kvCacheHit ? "true" : "false") << "\n"
      << "kv_restored_from_cold_store: "
      << (result.kvRestoredFromColdStore ? "true" : "false") << "\n"
      << "kv_page_count: " << result.kvPageCount << "\n"
      << "kv_hot_pages: " << result.kvHotPages << "\n"
      << "kv_warm_pages: " << result.kvWarmPages << "\n"
      << "kv_cold_pages: " << result.kvColdPages << "\n"
      << "kv_summary_rows: " << result.kvSummaryRows << "\n"
      << "prefix_cache_entries: " << result.prefixCacheEntries << "\n"
      << "mlx_plan_built: " << (result.mlxPlanBuilt ? "true" : "false") << "\n"
      << "mlx_evaluated: " << (result.mlxEvaluated ? "true" : "false") << "\n"
      << "moe_selected_experts: " << result.moeSelectedExperts << "\n"
      << "moe_router_entropy: " << result.moeRouterEntropy << "\n"
      << "moe_load_balance: " << result.moeLoadBalance << "\n"
      << "moe_selected_mass: " << result.moeSelectedMass << "\n"
      << "moe_pager_loads: " << result.moePagerLoads << "\n"
      << "moe_pager_evictions: " << result.moePagerEvictions << "\n"
      << "moe_pager_reuses: " << result.moePagerReuses << "\n"
      << "moe_resident_experts: " << result.moeResidentExperts << "\n"
      << "moe_prefetch_prefetched: " << result.moePrefetchPrefetched << "\n"
      << "moe_prefetch_hits: " << result.moePrefetchHits << "\n"
      << "moe_prefetch_misses: " << result.moePrefetchMisses << "\n"
      << "moe_prefetch_hit_rate: " << result.moePrefetchHitRatio << "\n"
      << "moe_prefetch_wrong_expert_leak_prevented: "
      << (result.moePrefetchWrongExpertLeakPrevented ? "true" : "false") << "\n"
      << "moe_prefetch_executable_experts: "
      << result.moePrefetchExecutableExperts << "\n"
      << "moe_sparsity_cache_hit: "
      << (result.moeSparsityCacheHit ? "true" : "false") << "\n"
      << "moe_sparsity_cache_hits: " << result.moeSparsityCacheHits << "\n"
      << "moe_sparsity_cache_misses: " << result.moeSparsityCacheMisses << "\n"
      << "moe_sparsity_cache_entries: " << result.moeSparsityCacheEntries
      << "\n"
      << "moe_sparsity_cache_hit_rate: "
      << ComputeMoeSparsityCacheHitRate(result) << "\n"
      << "moe_sparsity_pattern_hash: " << result.moeSparsityPatternHash << "\n"
      << "moe_sparsity_pattern_key: " << result.moeSparsityPatternKey << "\n"
      << "multimodal_cache_hit: "
      << (result.multimodalCacheHit ? "true" : "false") << "\n"
      << "multimodal_cache_hits: " << result.multimodalCacheHits << "\n"
      << "multimodal_cache_misses: " << result.multimodalCacheMisses << "\n"
      << "multimodal_cache_entries: " << result.multimodalCacheEntries << "\n"
      << "multimodal_cache_hit_rate: " << ComputeMultimodalCacheHitRate(result)
      << "\n"
      << "multimodal_active_modalities: " << result.multimodalActiveModalities
      << "\n"
      << "multimodal_modalities: " << result.multimodalModalities << "\n"
      << "moe_shard_count: " << result.moeShardCount << "\n"
      << "moe_active_experts: " << result.moeActiveExperts << "\n"
      << "moe_lazy_load: " << (result.moeLazyLoad ? "true" : "false") << "\n"
      << "shared_tokenizer: " << (result.sharedTokenizer ? "true" : "false")
      << "\n"
      << "speculative_accepted_tokens: " << result.speculativeAcceptedTokens
      << "\n"
      << "speculative_rejected_tokens: " << result.speculativeRejectedTokens
      << "\n"
      << "speculative_acceptance_rate: " << result.speculativeAcceptanceRate
      << "\n"
      << "speculative_fallback_token: " << result.speculativeFallbackToken
      << "\n"
      << "moe_hit_rate: " << ComputeMoeHitRate(result) << "\n"
      << "moe_eviction_rate: " << ComputeMoeEvictionRate(result) << "\n"
      << "weight_dtype: " << result.weightDType << "\n"
      << "neon_kernel_flavor: " << result.neonKernelFlavor << "\n"
      << "dequant_path: " << result.dequantPath << "\n"
      << "mixed_dispatch_strategy: " << result.mixedDispatchStrategy << "\n"
      << "mixed_dispatch_metal_stages: " << result.mixedDispatchMetalStages
      << "\n"
      << "mixed_dispatch_ane_stages: " << result.mixedDispatchAneStages << "\n"
      << "ane_compiled_layers: " << result.aneCompiledLayers << "\n"
      << "ane_prediction_calls: " << result.anePredictionCalls << "\n"
      << "thermal_pressure_level: " << result.thermalPressureLevel << "\n"
      << "thermal_reason: " << result.thermalReason << "\n"
      << "thermal_downgraded: " << (result.thermalDowngraded ? "true" : "false")
      << "\n"
      << "metal_device: " << result.metalDevice << "\n"
      << "metal_queue_label: " << result.metalQueueLabel << "\n"
      << "prompt_tokens: " << result.promptTokens.size() << "\n"
      << "generated_tokens: " << result.generatedTokens.size() << "\n"
      << "text: " << result.text << "\n";
}

void PrintRunJson(const us4::GenerationResult &result) {
  std::ostringstream promptTokens;
  std::ostringstream generatedTokens;

  for (std::size_t index = 0; index < result.promptTokens.size(); ++index) {
    if (index > 0) {
      promptTokens << ",";
    }
    promptTokens << "\"" << EscapeJson(result.promptTokens[index]) << "\"";
  }

  for (std::size_t index = 0; index < result.generatedTokens.size(); ++index) {
    if (index > 0) {
      generatedTokens << ",";
    }
    generatedTokens << "\"" << EscapeJson(result.generatedTokens[index])
                    << "\"";
  }

  std::cout
      << "{"
      << "\"family\":\"" << EscapeJson(result.family) << "\","
      << "\"model\":\"" << EscapeJson(result.modelName) << "\","
      << "\"asset_format\":\"" << EscapeJson(result.assetFormat) << "\","
      << "\"asset_path\":\"" << EscapeJson(result.assetPath) << "\","
      << "\"draft_model_format\":\"" << EscapeJson(result.draftModelFormat)
      << "\","
      << "\"draft_model_path\":\"" << EscapeJson(result.draftModelPath) << "\","
      << "\"speculative_strategy\":\"" << EscapeJson(result.speculativeStrategy)
      << "\","
      << "\"speculative_session_scope\":\""
      << EscapeJson(result.speculativeSessionScope) << "\","
      << "\"mode\":\"" << EscapeJson(us4::ToString(result.mode)) << "\","
      << "\"backend\":\"" << EscapeJson(result.backend) << "\","
      << "\"backend_reason\":\"" << EscapeJson(result.backendReason) << "\","
      << "\"fallback\":" << (result.fellBack ? "true" : "false") << ","
      << "\"shared_allocations\":" << result.sharedAllocations << ","
      << "\"metal_dispatches\":" << result.metalDispatches << ","
      << "\"mlx_operation_count\":" << result.mlxOperationCount << ","
      << "\"kv_cache_hit\":" << (result.kvCacheHit ? "true" : "false") << ","
      << "\"kv_restored_from_cold_store\":"
      << (result.kvRestoredFromColdStore ? "true" : "false") << ","
      << "\"kv_page_count\":" << result.kvPageCount << ","
      << "\"kv_hot_pages\":" << result.kvHotPages << ","
      << "\"kv_warm_pages\":" << result.kvWarmPages << ","
      << "\"kv_cold_pages\":" << result.kvColdPages << ","
      << "\"kv_summary_rows\":" << result.kvSummaryRows << ","
      << "\"prefix_cache_entries\":" << result.prefixCacheEntries << ","
      << "\"mlx_plan_built\":" << (result.mlxPlanBuilt ? "true" : "false")
      << ","
      << "\"mlx_evaluated\":" << (result.mlxEvaluated ? "true" : "false") << ","
      << "\"moe_selected_experts\":" << result.moeSelectedExperts << ","
      << "\"moe_router_entropy\":" << result.moeRouterEntropy << ","
      << "\"moe_load_balance\":" << result.moeLoadBalance << ","
      << "\"moe_selected_mass\":" << result.moeSelectedMass << ","
      << "\"moe_pager_loads\":" << result.moePagerLoads << ","
      << "\"moe_pager_evictions\":" << result.moePagerEvictions << ","
      << "\"moe_pager_reuses\":" << result.moePagerReuses << ","
      << "\"moe_resident_experts\":" << result.moeResidentExperts << ","
      << "\"moe_prefetch_prefetched\":" << result.moePrefetchPrefetched << ","
      << "\"moe_prefetch_hits\":" << result.moePrefetchHits << ","
      << "\"moe_prefetch_misses\":" << result.moePrefetchMisses << ","
      << "\"moe_prefetch_hit_rate\":" << result.moePrefetchHitRatio << ","
      << "\"moe_prefetch_wrong_expert_leak_prevented\":"
      << (result.moePrefetchWrongExpertLeakPrevented ? "true" : "false") << ","
      << "\"moe_prefetch_executable_experts\":"
      << result.moePrefetchExecutableExperts << ","
      << "\"moe_sparsity_cache_hit\":"
      << (result.moeSparsityCacheHit ? "true" : "false") << ","
      << "\"moe_sparsity_cache_hits\":" << result.moeSparsityCacheHits << ","
      << "\"moe_sparsity_cache_misses\":" << result.moeSparsityCacheMisses
      << ","
      << "\"moe_sparsity_cache_entries\":" << result.moeSparsityCacheEntries
      << ","
      << "\"moe_sparsity_cache_hit_rate\":"
      << ComputeMoeSparsityCacheHitRate(result) << ","
      << "\"moe_sparsity_pattern_hash\":" << result.moeSparsityPatternHash
      << ","
      << "\"moe_sparsity_pattern_key\":\""
      << EscapeJson(result.moeSparsityPatternKey) << "\","
      << "\"multimodal_cache_hit\":"
      << (result.multimodalCacheHit ? "true" : "false") << ","
      << "\"multimodal_cache_hits\":" << result.multimodalCacheHits << ","
      << "\"multimodal_cache_misses\":" << result.multimodalCacheMisses << ","
      << "\"multimodal_cache_entries\":" << result.multimodalCacheEntries << ","
      << "\"multimodal_cache_hit_rate\":"
      << ComputeMultimodalCacheHitRate(result) << ","
      << "\"multimodal_active_modalities\":"
      << result.multimodalActiveModalities << ","
      << "\"multimodal_modalities\":\""
      << EscapeJson(result.multimodalModalities) << "\","
      << "\"moe_shard_count\":" << result.moeShardCount << ","
      << "\"moe_active_experts\":" << result.moeActiveExperts << ","
      << "\"moe_lazy_load\":" << (result.moeLazyLoad ? "true" : "false") << ","
      << "\"shared_tokenizer\":" << (result.sharedTokenizer ? "true" : "false")
      << ","
      << "\"speculative_accepted_tokens\":" << result.speculativeAcceptedTokens
      << ","
      << "\"speculative_rejected_tokens\":" << result.speculativeRejectedTokens
      << ","
      << "\"speculative_acceptance_rate\":" << result.speculativeAcceptanceRate
      << ","
      << "\"speculative_fallback_token\":\""
      << EscapeJson(result.speculativeFallbackToken) << "\","
      << "\"moe_hit_rate\":" << ComputeMoeHitRate(result) << ","
      << "\"moe_eviction_rate\":" << ComputeMoeEvictionRate(result) << ","
      << "\"weight_dtype\":\"" << EscapeJson(result.weightDType) << "\","
      << "\"neon_kernel_flavor\":\"" << EscapeJson(result.neonKernelFlavor)
      << "\","
      << "\"dequant_path\":\"" << EscapeJson(result.dequantPath) << "\","
      << "\"mixed_dispatch_strategy\":\""
      << EscapeJson(result.mixedDispatchStrategy) << "\","
      << "\"mixed_dispatch_metal_stages\":" << result.mixedDispatchMetalStages
      << ","
      << "\"mixed_dispatch_ane_stages\":" << result.mixedDispatchAneStages
      << ","
      << "\"ane_compiled_layers\":" << result.aneCompiledLayers << ","
      << "\"ane_prediction_calls\":" << result.anePredictionCalls << ","
      << "\"thermal_pressure_level\":\""
      << EscapeJson(result.thermalPressureLevel) << "\","
      << "\"thermal_reason\":\"" << EscapeJson(result.thermalReason) << "\","
      << "\"thermal_downgraded\":"
      << (result.thermalDowngraded ? "true" : "false") << ","
      << "\"metal_device\":\"" << EscapeJson(result.metalDevice) << "\","
      << "\"metal_queue_label\":\"" << EscapeJson(result.metalQueueLabel)
      << "\","
      << "\"prompt_tokens\":[" << promptTokens.str() << "],"
      << "\"generated_tokens\":[" << generatedTokens.str() << "],"
      << "\"text\":\"" << EscapeJson(result.text) << "\""
      << "}\n";
}

void PrintAdapterList(const us4::HardwareProbeResult &probe) {
  std::cout << "Available models:\n";
  for (const us4::IUS4V6Adapter *adapter : us4::ListAdapters()) {
    const us4::BackendSelection preferred =
        us4::SelectBackend(probe, probe.recommendedMode, *adapter);
    std::cout << "  - " << adapter->ModelName() << " [" << adapter->Family()
              << "]"
              << " arch=" << ArchitectureToString(adapter->Architecture())
              << " min_mode=" << us4::ToString(adapter->MinimumMode())
              << " mlx=" << (adapter->SupportsMlxBackend() ? "true" : "false")
              << " metal="
              << (adapter->SupportsMetalBackend() ? "true" : "false")
              << " moe=" << (adapter->SupportsMoe() ? "true" : "false")
              << " preferred_backend=" << us4::ToString(preferred.selected)
              << "\n";
  }
}

void PrintAdapterListJson(const us4::HardwareProbeResult &probe) {
  std::ostringstream models;
  bool first = true;
  for (const us4::IUS4V6Adapter *adapter : us4::ListAdapters()) {
    const us4::BackendSelection preferred =
        us4::SelectBackend(probe, probe.recommendedMode, *adapter);
    if (!first) {
      models << ",";
    }
    models << "{"
           << "\"family\":\"" << EscapeJson(adapter->Family()) << "\","
           << "\"model\":\"" << EscapeJson(adapter->ModelName()) << "\","
           << "\"architecture\":\""
           << EscapeJson(ArchitectureToString(adapter->Architecture())) << "\","
           << "\"minimum_mode\":\""
           << EscapeJson(us4::ToString(adapter->MinimumMode())) << "\","
           << "\"supports_moe\":" << (adapter->SupportsMoe() ? "true" : "false")
           << ","
           << "\"supports_mlx\":"
           << (adapter->SupportsMlxBackend() ? "true" : "false") << ","
           << "\"supports_metal\":"
           << (adapter->SupportsMetalBackend() ? "true" : "false") << ","
           << "\"supports_prompt_run\":"
           << (adapter->SupportsPromptRun() ? "true" : "false") << ","
           << "\"preferred_backend\":\""
           << EscapeJson(us4::ToString(preferred.selected)) << "\","
           << "\"preferred_backend_reason\":\"" << EscapeJson(preferred.reason)
           << "\","
           << "\"preferred_mode\":\""
           << EscapeJson(us4::ToString(probe.recommendedMode)) << "\""
           << "}";
    first = false;
  }
  std::cout << "{\"models\":[" << models.str() << "]}\n";
}

} // namespace

int main(int argc, char **argv) {
  bool outputJson = false;
  bool showProbe = false;
  bool showVersion = false;
  bool showHelp = false;
  bool listModels = false;
  bool runCommand = false;
  std::optional<std::string> modeValue;
  std::optional<std::string> modelName;
  std::optional<std::string> modelPath;
  std::optional<std::string> backendValue;
  std::optional<std::string> promptValue;
  std::size_t maxTokens = 16;

  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "run") {
      runCommand = true;
    } else if (arg == "list-models") {
      listModels = true;
    } else if (arg == "--json") {
      outputJson = true;
    } else if (arg == "--probe") {
      showProbe = true;
    } else if (arg == "--version") {
      showVersion = true;
    } else if (arg == "--help" || arg == "-h") {
      showHelp = true;
    } else if (arg == "--mode" && index + 1 < argc) {
      modeValue = argv[++index];
    } else if (arg == "--model" && index + 1 < argc) {
      modelName = argv[++index];
    } else if (arg == "--model-path" && index + 1 < argc) {
      modelPath = argv[++index];
    } else if (arg == "--backend" && index + 1 < argc) {
      backendValue = argv[++index];
    } else if (arg == "--prompt" && index + 1 < argc) {
      promptValue = argv[++index];
    } else if (arg == "--max-tokens" && index + 1 < argc) {
      const std::string tokenText = argv[++index];
      try {
        const unsigned long long parsed = std::stoull(tokenText);
        maxTokens = static_cast<std::size_t>(std::min<unsigned long long>(
            parsed, static_cast<unsigned long long>(
                        std::numeric_limits<std::size_t>::max())));
      } catch (...) {
        std::cerr << "Invalid --max-tokens value: " << tokenText << "\n";
        return 1;
      }
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintHelp();
      return 1;
    }
  }

  if (showHelp || argc == 1) {
    PrintHelp();
    return 0;
  }

  if (showVersion) {
    std::cout << us4::kUs4Version << "\n";
    return 0;
  }

  const us4::HardwareProbeResult probe = us4::HardwareProbe::Detect();

  if (showProbe) {
    if (outputJson) {
      PrintProbeJson(probe);
    } else {
      PrintProbeText(probe);
    }
    return 0;
  }

  if (modeValue.has_value() && !runCommand) {
    const auto parsedMode = us4::ParseRuntimeMode(*modeValue);
    const us4::RuntimeMode mode =
        (*modeValue == "auto" || !parsedMode.has_value())
            ? probe.recommendedMode
            : *parsedMode;

    if (outputJson) {
      std::cout << "{\"mode\":\"" << us4::ToString(mode) << "\"}\n";
    } else {
      std::cout << us4::ToString(mode) << "\n";
    }
    return 0;
  }

  if (runCommand) {
    const std::optional<us4::BackendType> parsedBackend =
        backendValue.has_value() ? us4::ParseBackendType(*backendValue)
                                 : std::nullopt;
    if (backendValue.has_value() && !parsedBackend.has_value()) {
      std::cerr << "Invalid --backend value: " << *backendValue << "\n";
      return 1;
    }

    std::optional<us4::ModelAsset> loadedAsset;
    if (modelPath.has_value()) {
      us4::ModelAsset asset;
      std::string error;
      if (!us4::LoadModelAsset(std::filesystem::path(*modelPath), asset,
                               &error)) {
        std::cerr << "Failed to load --model-path: " << error << "\n";
        return 1;
      }
      loadedAsset = asset;
    }

    if (!modelName.has_value() &&
        (!loadedAsset.has_value() || loadedAsset->modelName.empty())) {
      std::cerr << "--model is required for run\n";
      PrintAdapterList(probe);
      return 1;
    }

    const std::string resolvedModelName = modelName.value_or(
        loadedAsset.has_value() ? loadedAsset->modelName : std::string{});
    const us4::IUS4V6Adapter *adapter =
        us4::FindAdapterByModel(resolvedModelName);
    if (adapter == nullptr) {
      std::cerr << "Unknown model: " << resolvedModelName << "\n";
      PrintAdapterList(probe);
      return 1;
    }

    us4::RuntimeContext context(probe);
    if (modeValue.has_value()) {
      const auto parsedMode = us4::ParseRuntimeMode(*modeValue);
      context.SetMode(parsedMode.has_value() ? *parsedMode
                                             : probe.recommendedMode);
    } else {
      adapter->ConfigureRuntime(context);
    }

    const us4::GenerationRequest request{
        .prompt = promptValue.value_or("hi"),
        .maxTokens = maxTokens,
        .asset = loadedAsset.has_value() ? &*loadedAsset : nullptr,
        .requestedBackend = parsedBackend,
    };
    const us4::GenerationResult result = adapter->Generate(request, context);
    if (outputJson) {
      PrintRunJson(result);
    } else {
      PrintRunText(result);
    }
    return 0;
  }

  if (listModels) {
    if (outputJson) {
      PrintAdapterListJson(probe);
    } else {
      PrintAdapterList(probe);
    }
    return 0;
  }

  PrintHelp();
  return 0;
}
