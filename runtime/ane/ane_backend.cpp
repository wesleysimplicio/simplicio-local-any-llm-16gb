#include "ane/ane_backend.h"

namespace us4 {

std::string_view ToString(const AneModelKind kind) {
  switch (kind) {
  case AneModelKind::kDenseProjection:
    return "dense-projection";
  case AneModelKind::kAttentionMlp:
    return "attention-mlp";
  }
  return "dense-projection";
}

AneBackend::AneBackend(const HardwareProbeResult &hardware)
    : available_(hardware.hasAne) {
  if (!hardware.hasAne) {
    reason_ = "ane-unavailable";
    return;
  }

  reason_ = hardware.chip.find("M5") != std::string::npos ? "ane-backend-ready"
                                                          : "ane-host-ready";
}

bool AneBackend::Available() const { return available_; }

std::string_view AneBackend::Reason() const { return reason_; }

bool AneBackend::Compile(const AneCompilePlan &plan) {
  if (!available_ || plan.family.empty() || plan.layerName.empty() ||
      plan.tokenCount == 0) {
    return false;
  }

  lastCompiledModel_ = AneCompiledModel{
      .kind = plan.kind,
      .family = plan.family,
      .layerName = plan.layerName,
      .computeUnits = "ane-only",
      .targetChip = "apple-m5+",
      .usedCoreMlCompileIntent = true,
      .supportsPrediction = true,
      .staticShapePreferred = plan.staticShapePreferred,
  };
  lastPredictionSucceeded_ = false;
  reason_ = "ane-model-compiled";
  return true;
}

bool AneBackend::Predict(const std::size_t acceptedTokens,
                         const std::size_t rejectedTokens) {
  if (!available_ || !lastCompiledModel_.has_value() ||
      !lastCompiledModel_->supportsPrediction) {
    return false;
  }

  if (acceptedTokens == 0 && rejectedTokens == 0) {
    return false;
  }

  ++predictionCount_;
  lastPredictionSucceeded_ = true;
  reason_ = "ane-predict-recorded";
  return true;
}

bool AneBackend::LastPredictionSucceeded() const {
  return lastPredictionSucceeded_;
}

std::size_t AneBackend::PredictionCount() const { return predictionCount_; }

const std::optional<AneCompiledModel> &AneBackend::LastCompiledModel() const {
  return lastCompiledModel_;
}

} // namespace us4
