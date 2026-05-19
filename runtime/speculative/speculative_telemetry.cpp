#include "speculative/speculative_telemetry.h"

namespace us4 {

SpeculativeTelemetry ComputeSpeculativeTelemetry(const std::size_t draftAttempts,
                                                 const std::size_t acceptedTokens,
                                                 const std::size_t verifyWindow) {
  SpeculativeTelemetry telemetry;
  telemetry.draftAttempts = draftAttempts;
  telemetry.acceptedTokens = acceptedTokens;
  telemetry.rejectedTokens = draftAttempts > acceptedTokens
                                 ? draftAttempts - acceptedTokens
                                 : 0;
  telemetry.verifyWindow = verifyWindow;
  if (draftAttempts > 0) {
    telemetry.acceptanceRate =
        static_cast<float>(acceptedTokens) / static_cast<float>(draftAttempts);
  }
  return telemetry;
}

} // namespace us4
