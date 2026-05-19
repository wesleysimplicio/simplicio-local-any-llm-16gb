#include "kv/eviction_policy.h"

#include <algorithm>

namespace us4 {

EvictionDecision SelectEvictionVictim(const std::vector<EvictionCandidate> &candidates) {
  EvictionDecision decision;
  if (candidates.empty()) {
    return decision;
  }

  auto picked = std::min_element(
      candidates.begin(), candidates.end(),
      [](const EvictionCandidate &lhs, const EvictionCandidate &rhs) {
        if (lhs.hitCount != rhs.hitCount) {
          return lhs.hitCount < rhs.hitCount;
        }
        return lhs.lastTouch < rhs.lastTouch;
      });

  decision.key = picked->key;
  decision.valid = true;
  if (picked->hitCount == 0) {
    decision.reason = "cold";
  } else {
    decision.reason = "lru-frequency";
  }
  return decision;
}

} // namespace us4
