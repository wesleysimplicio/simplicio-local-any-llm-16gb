#include "moe/router_metrics.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace us4 {

namespace {

std::vector<float> Softmax(const std::vector<float> &logits) {
  if (logits.empty()) {
    return {};
  }
  const float maxLogit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> probs(logits.size(), 0.0F);
  float denominator = 0.0F;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    probs[i] = std::exp(logits[i] - maxLogit);
    denominator += probs[i];
  }
  if (denominator <= 0.0F) {
    std::fill(probs.begin(), probs.end(), 1.0F / static_cast<float>(probs.size()));
    return probs;
  }
  for (auto &value : probs) {
    value /= denominator;
  }
  return probs;
}

} // namespace

float RouterEntropy(const std::vector<float> &logits) {
  const auto probs = Softmax(logits);
  if (probs.empty()) {
    return 0.0F;
  }
  float entropy = 0.0F;
  for (const float probability : probs) {
    if (probability > 0.0F) {
      entropy -= probability * std::log(probability);
    }
  }
  return entropy;
}

float RouterLoadBalanceLoss(const std::vector<float> &logits,
                            const std::size_t k) {
  if (logits.empty() || k == 0) {
    return 0.0F;
  }
  const auto probs = Softmax(logits);
  const float idealLoad =
      static_cast<float>(k) / static_cast<float>(probs.size());
  float loss = 0.0F;
  for (const float probability : probs) {
    const float gap = probability - idealLoad;
    loss += gap * gap;
  }
  return loss;
}

RoutingTelemetry ComputeRoutingTelemetry(const std::vector<float> &logits,
                                         const std::size_t k) {
  RoutingTelemetry telemetry;
  telemetry.consideredExperts = logits.size();
  if (logits.empty() || k == 0) {
    return telemetry;
  }
  Router router;
  telemetry.topK = router.TopK(logits, k);
  telemetry.entropy = RouterEntropy(logits);
  telemetry.loadBalanceLoss = RouterLoadBalanceLoss(logits, k);
  return telemetry;
}

} // namespace us4
