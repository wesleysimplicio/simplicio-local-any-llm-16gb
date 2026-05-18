#include "scheduler/continuous_batcher.h"

#include <algorithm>
#include <utility>

namespace us4 {

ContinuousBatcher::ContinuousBatcher(const std::size_t maxBatchTokens)
    : maxBatchTokens_(std::max<std::size_t>(1U, maxBatchTokens)) {}

std::size_t ContinuousBatcher::MaxBatchTokens() const noexcept {
  return maxBatchTokens_;
}

BatchDecision
ContinuousBatcher::Schedule(const std::vector<SessionDemand> &demands) const {
  BatchDecision decision;
  decision.totalGrantedTokens = 0U;
  decision.activeSessions = 0U;
  decision.fairnessRounds = 0U;
  decision.singleSessionPassthrough = false;

  struct ActiveDemand {
    SessionDemand demand;
    std::size_t remainingTokens = 0U;
    std::size_t grantedTokens = 0U;
    std::size_t roundsVisited = 0U;
  };

  std::vector<ActiveDemand> active;
  active.reserve(demands.size());
  for (const SessionDemand &demand : demands) {
    if (demand.pendingTokens == 0U) {
      continue;
    }
    ActiveDemand state;
    state.demand = demand;
    state.remainingTokens = demand.pendingTokens;
    state.grantedTokens = 0U;
    state.roundsVisited = 0U;
    active.push_back(std::move(state));
  }

  std::sort(active.begin(), active.end(),
            [](const ActiveDemand &lhs, const ActiveDemand &rhs) {
              if (lhs.demand.arrivalOrder != rhs.demand.arrivalOrder) {
                return lhs.demand.arrivalOrder < rhs.demand.arrivalOrder;
              }
              return lhs.demand.sessionId < rhs.demand.sessionId;
            });

  decision.activeSessions = active.size();
  decision.singleSessionPassthrough = active.size() == 1U;

  if (decision.singleSessionPassthrough && !active.empty()) {
    ActiveDemand &state = active.front();
    const std::size_t grant = std::min(state.remainingTokens, maxBatchTokens_);
    state.remainingTokens -= grant;
    state.grantedTokens = grant;
    state.roundsVisited = grant > 0U ? 1U : 0U;
    decision.totalGrantedTokens = grant;
    decision.fairnessRounds = grant > 0U ? 1U : 0U;
    decision.slices.push_back(
        {state.demand.sessionId, state.grantedTokens, state.roundsVisited});
    return decision;
  }

  std::size_t remainingBatchTokens = maxBatchTokens_;
  while (remainingBatchTokens > 0U) {
    bool anyGrantThisRound = false;
    for (ActiveDemand &state : active) {
      if (remainingBatchTokens == 0U) {
        break;
      }
      if (state.remainingTokens == 0U) {
        continue;
      }

      const std::size_t fairnessWeight =
          std::max<std::size_t>(1U, state.demand.fairnessWeight);
      const std::size_t grant = std::min(
          {state.remainingTokens, fairnessWeight, remainingBatchTokens});
      if (grant == 0U) {
        continue;
      }

      state.remainingTokens -= grant;
      state.grantedTokens += grant;
      state.roundsVisited += 1U;
      remainingBatchTokens -= grant;
      decision.totalGrantedTokens += grant;
      anyGrantThisRound = true;
    }

    if (!anyGrantThisRound) {
      break;
    }
    decision.fairnessRounds += 1U;
  }

  decision.slices.reserve(active.size());
  for (const ActiveDemand &state : active) {
    if (state.grantedTokens == 0U) {
      continue;
    }
    decision.slices.push_back(
        {state.demand.sessionId, state.grantedTokens, state.roundsVisited});
  }

  return decision;
}

} // namespace us4
