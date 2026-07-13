#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include "core/backend_selector.h"
#include "neon/kernel_profile.h"

namespace us4::benchmarks {

inline std::string_view BackendLabel(const BackendType backend) {
  switch (backend) {
  case BackendType::kScalarCpu:
    return "scalar";
  case BackendType::kNeon:
    return "neon";
  case BackendType::kMlx:
    return "mlx";
  case BackendType::kMetal:
    return "metal";
  case BackendType::kAne:
    return "ane";
  }
  return "scalar";
}

struct KernelObservation {
  std::string_view requestedBackend = "auto";
  std::string_view observedBackend = "scalar";
  std::string_view backendReason = "requested";
  std::string_view kernelFlavor = "scalar-bridge";
};

inline KernelObservation ObserveKernelMatmul(
    const std::optional<BackendType> requestedBackend,
    const NeonMatmulProfile &profile) {
  if (!requestedBackend.has_value()) {
    return {
        .requestedBackend = "auto",
        .observedBackend =
            profile.flavor == NeonKernelFlavor::kScalarBridge ? "scalar"
                                                              : "neon",
        .backendReason =
            profile.flavor == NeonKernelFlavor::kScalarBridge ? "auto-scalar"
                                                              : "auto-neon",
        .kernelFlavor = ToString(profile.flavor),
    };
  }

  if (*requestedBackend == BackendType::kScalarCpu) {
    return {
        .requestedBackend = "scalar",
        .observedBackend = "scalar",
        .backendReason = "requested",
        .kernelFlavor = "scalar-reference",
    };
  }

  const bool neonExecuted = profile.flavor != NeonKernelFlavor::kScalarBridge;
  return {
      .requestedBackend = BackendLabel(*requestedBackend),
      .observedBackend = neonExecuted ? "neon" : "scalar",
      .backendReason = neonExecuted ? "requested"
                                    : "requested-backend-unavailable",
      .kernelFlavor = ToString(profile.flavor),
  };
}

inline KernelObservation ObserveKernelAttention(
    const std::optional<BackendType> requestedBackend,
    const NeonAttentionProfile &profile) {
  if (!requestedBackend.has_value()) {
    return {
        .requestedBackend = "auto",
        .observedBackend =
            profile.flavor == NeonKernelFlavor::kScalarBridge ? "scalar"
                                                              : "neon",
        .backendReason =
            profile.flavor == NeonKernelFlavor::kScalarBridge ? "auto-scalar"
                                                              : "auto-neon",
        .kernelFlavor = ToString(profile.flavor),
    };
  }

  if (*requestedBackend == BackendType::kScalarCpu) {
    return {
        .requestedBackend = "scalar",
        .observedBackend = "scalar",
        .backendReason = "requested",
        .kernelFlavor = "scalar-reference",
    };
  }

  const bool neonExecuted = profile.flavor != NeonKernelFlavor::kScalarBridge;
  return {
      .requestedBackend = BackendLabel(*requestedBackend),
      .observedBackend = neonExecuted ? "neon" : "scalar",
      .backendReason = neonExecuted ? "requested"
                                    : "requested-backend-unavailable",
      .kernelFlavor = ToString(profile.flavor),
  };
}

inline double MatmulOperations(const std::size_t lhsRows,
                               const std::size_t lhsCols,
                               const std::size_t rhsCols) {
  return static_cast<double>(2ULL) * static_cast<double>(lhsRows) *
         static_cast<double>(lhsCols) * static_cast<double>(rhsCols);
}

inline double AttentionOperations(const std::size_t totalVisibleRows,
                                  const std::size_t hidden,
                                  const std::size_t valueWidth) {
  return static_cast<double>(2ULL) * static_cast<double>(totalVisibleRows) *
         static_cast<double>(hidden + valueWidth);
}

inline double ThroughputGops(const double operations, const double elapsedMs) {
  if (elapsedMs <= 0.0) {
    return 0.0;
  }
  return operations / (elapsedMs / 1000.0) / 1'000'000'000.0;
}

} // namespace us4::benchmarks
