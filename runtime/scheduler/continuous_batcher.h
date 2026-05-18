#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace us4 {

struct SessionDemand {
  std::string sessionId;
  std::size_t pendingTokens = 0U;
  std::size_t fairnessWeight = 1U;
  std::size_t arrivalOrder = 0U;
};

struct ScheduledSlice {
  std::string sessionId;
  std::size_t grantedTokens = 0U;
  std::size_t roundsVisited = 0U;
};

struct BatchDecision {
  std::vector<ScheduledSlice> slices;
  std::size_t totalGrantedTokens = 0U;
  std::size_t activeSessions = 0U;
  std::size_t fairnessRounds = 0U;
  bool singleSessionPassthrough = false;
};

class ContinuousBatcher {
public:
  explicit ContinuousBatcher(std::size_t maxBatchTokens = 8U);

  [[nodiscard]] std::size_t MaxBatchTokens() const noexcept;
  [[nodiscard]] BatchDecision
  Schedule(const std::vector<SessionDemand> &demands) const;

private:
  std::size_t maxBatchTokens_ = 8U;
};

} // namespace us4
