#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "core/backend_selector.h"
#include "core/ius4v6_adapter.h"

namespace us4::benchmarks {

struct CaseObservation {
  std::string label;
  std::string requestedBackend;
  std::string observedBackend;
  std::string backendReason;
  std::string weightDType;
  std::string neonKernelFlavor;
  std::string dequantPath;
  std::size_t generatedTokenCount = 0;
  std::string text;
  std::uint64_t textFingerprint = 0;
  long long elapsedMs = 0;
  bool fellBack = false;
};

struct LowBitRegression {
  bool comparable = false;
  bool textMatch = false;
  bool tokenCountMatch = false;
  bool dequantPathMatch = false;
  bool neonKernelVisible = false;
  bool neonExecuted = false;
  bool fallbackObserved = false;
  double speedupVsScalar = 0.0;
  std::string status = "fail";
  std::string reason = "uninitialized";
};

inline std::string RequestedBackendLabel(
    const std::optional<BackendType> requestedBackend) {
  return requestedBackend.has_value() ? std::string(ToString(*requestedBackend))
                                      : "auto";
}

inline std::uint64_t FingerprintText(const std::string_view text) {
  constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

  std::uint64_t hash = kFnvOffset;
  for (const char ch : text) {
    hash ^= static_cast<unsigned char>(ch);
    hash *= kFnvPrime;
  }
  return hash;
}

inline std::string
ExpectedLowBitDequantPath(const std::string_view weightDType) {
  if (weightDType == "int8") {
    return "groupwise-int8";
  }
  if (weightDType == "int4") {
    return "groupwise-int4";
  }
  return "none";
}

inline bool HasVisibleLowBitKernel(const CaseObservation &observation) {
  if (observation.weightDType == "int8") {
    return observation.neonKernelFlavor == "int8-dot";
  }
  if (observation.weightDType == "int4") {
    return !observation.neonKernelFlavor.empty() &&
           observation.neonKernelFlavor != "none";
  }
  return !observation.neonKernelFlavor.empty() &&
         observation.neonKernelFlavor != "none";
}

inline CaseObservation ObserveCase(const std::string_view label,
                                   const GenerationResult &result,
                                   const long long elapsedMs,
                                   const std::optional<BackendType>
                                       requestedBackend = std::nullopt) {
  CaseObservation observation;
  observation.label = std::string(label);
  observation.requestedBackend = RequestedBackendLabel(requestedBackend);
  observation.observedBackend = result.backend;
  observation.backendReason = result.backendReason;
  observation.weightDType = result.weightDType;
  observation.neonKernelFlavor = result.neonKernelFlavor;
  observation.dequantPath = result.dequantPath;
  observation.generatedTokenCount = result.generatedTokens.size();
  observation.text = result.text;
  observation.textFingerprint = FingerprintText(result.text);
  observation.elapsedMs = elapsedMs;
  observation.fellBack = result.fellBack;
  return observation;
}

inline LowBitRegression CompareLowBitObservations(
    const CaseObservation &scalarObservation,
    const CaseObservation &neonObservation) {
  LowBitRegression regression;
  regression.comparable = scalarObservation.weightDType ==
                              neonObservation.weightDType &&
                          scalarObservation.generatedTokenCount > 0 &&
                          neonObservation.generatedTokenCount > 0;
  regression.textMatch =
      scalarObservation.textFingerprint == neonObservation.textFingerprint &&
      scalarObservation.text == neonObservation.text;
  regression.tokenCountMatch =
      scalarObservation.generatedTokenCount ==
      neonObservation.generatedTokenCount;

  const std::string expectedDequantPath =
      ExpectedLowBitDequantPath(neonObservation.weightDType);
  regression.dequantPathMatch =
      scalarObservation.dequantPath == expectedDequantPath &&
      neonObservation.dequantPath == expectedDequantPath;
  regression.neonKernelVisible = HasVisibleLowBitKernel(neonObservation);
  regression.neonExecuted =
      neonObservation.observedBackend == "neon" && !neonObservation.fellBack;
  regression.fallbackObserved =
      neonObservation.fellBack || neonObservation.observedBackend != "neon";

  if (scalarObservation.elapsedMs > 0 && neonObservation.elapsedMs > 0) {
    regression.speedupVsScalar =
        static_cast<double>(scalarObservation.elapsedMs) /
        static_cast<double>(neonObservation.elapsedMs);
  }

  if (!regression.comparable) {
    regression.status = "fail";
    regression.reason = "non-comparable-observations";
    return regression;
  }

  if (!regression.textMatch || !regression.tokenCountMatch ||
      !regression.dequantPathMatch ||
      (regression.neonExecuted && !regression.neonKernelVisible)) {
    regression.status = "fail";
    regression.reason = "output-or-metadata-drift";
    return regression;
  }

  if (regression.neonExecuted) {
    regression.status = "pass";
    regression.reason = "outputs-match-and-neon-path-visible";
    return regression;
  }

  regression.status = "warn";
  regression.reason = "requested-neon-fell-back-with-matching-output";
  return regression;
}

} // namespace us4::benchmarks
