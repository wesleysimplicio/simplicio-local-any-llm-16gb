#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "core/hardware_probe.h"

namespace us4 {

enum class AneModelKind {
  kDenseProjection,
  kAttentionMlp,
};

struct AneCompilePlan {
  AneModelKind kind = AneModelKind::kDenseProjection;
  std::string family;
  std::string layerName;
  std::size_t tokenCount = 0;
  bool usesSharedTokenizer = false;
  bool staticShapePreferred = true;
};

struct AneCompiledModel {
  AneModelKind kind = AneModelKind::kDenseProjection;
  std::string family;
  std::string layerName;
  std::string computeUnits;
  std::string targetChip;
  bool usedCoreMlCompileIntent = false;
  bool supportsPrediction = false;
  bool staticShapePreferred = true;
};

std::string_view ToString(AneModelKind kind);

class AneBackend {
public:
  AneBackend() = default;
  explicit AneBackend(const HardwareProbeResult &hardware);

  bool Available() const;
  std::string_view Reason() const;
  bool Compile(const AneCompilePlan &plan);
  bool Predict(std::size_t acceptedTokens, std::size_t rejectedTokens);
  bool LastPredictionSucceeded() const;
  std::size_t PredictionCount() const;
  const std::optional<AneCompiledModel> &LastCompiledModel() const;

private:
  bool available_ = false;
  bool lastPredictionSucceeded_ = false;
  std::size_t predictionCount_ = 0;
  std::string reason_ = "ane-unavailable";
  std::optional<AneCompiledModel> lastCompiledModel_;
};

} // namespace us4
