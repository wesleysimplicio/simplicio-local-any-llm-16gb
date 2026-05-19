#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace us4 {

// EvictionCandidate describes a single page seen by the eviction policy.
// hitCount and lastTouch come from `KvPage` instrumentation; the policy never
// mutates pages directly, it only ranks them.
struct EvictionCandidate {
  std::string key;
  std::size_t hitCount = 0;
  std::size_t lastTouch = 0;
};

// EvictionDecision returns the page key chosen for eviction plus a stable
// reason tag so telemetry can attribute the eviction in CLI/bench output.
struct EvictionDecision {
  std::string key;
  std::string reason;
  bool valid = false;
};

// Hybrid LRU + frequency policy. The cheapest page is the one with the lowest
// hit count; ties are broken by the oldest `lastTouch`. The policy is
// deterministic for stable inputs and exposes the reason it chose a candidate.
EvictionDecision SelectEvictionVictim(const std::vector<EvictionCandidate> &candidates);

} // namespace us4
