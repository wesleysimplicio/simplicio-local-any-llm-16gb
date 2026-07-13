#include "cache/sparsity_aware_cache.h"

#include <algorithm>
#include <utility>

namespace us4 {

namespace {

constexpr std::size_t kHashOffset = 1469598103934665603ULL;
constexpr std::size_t kHashPrime = 1099511628211ULL;

} // namespace

std::optional<SparsityCacheEntry>
SparsityAwareCache::Lookup(const std::string_view family,
                           const RouterDecision &routing) const {
  const std::size_t patternHash = ComputePatternHash(routing.selected);
  const std::string key = BuildKey(family, patternHash, routing.selected);
  const auto it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  return it->second;
}

SparsityCacheSnapshot SparsityAwareCache::Touch(const std::string_view family,
                                                const RouterDecision &routing) {
  const std::size_t patternHash = ComputePatternHash(routing.selected);
  const std::string key = BuildKey(family, patternHash, routing.selected);
  const auto it = entries_.find(key);
  if (it != entries_.end()) {
    ++hitCount_;
    ++it->second.uses;
    ++it->second.hits;
    if (it->second.hits >= 1U) {
      it->second.warm = true;
    }
    return Snapshot(true, key, patternHash, it->second.warm);
  }

  ++missCount_;
  SparsityCacheEntry entry;
  entry.family = std::string(family);
  entry.key = key;
  entry.patternHash = patternHash;
  entry.uses = 1U;
  entry.hits = 0U;
  entry.warm = false;
  entry.experts.reserve(routing.selected.size());
  for (const ExpertScore &expert : routing.selected) {
    entry.experts.push_back(expert.expert);
  }
  entries_.emplace(key, std::move(entry));
  return Snapshot(false, key, patternHash, false);
}

std::size_t SparsityAwareCache::EntryCount() const { return entries_.size(); }

std::size_t SparsityAwareCache::ComputePatternHash(
    const std::vector<ExpertScore> &experts) {
  std::size_t hash = kHashOffset;
  for (const ExpertScore &expert : experts) {
    hash ^= expert.expert + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    hash *= kHashPrime;
  }
  return hash;
}

std::string
SparsityAwareCache::BuildKey(const std::string_view family,
                             const std::size_t patternHash,
                             const std::vector<ExpertScore> &experts) {
  std::string key(family);
  key += ":";
  key += std::to_string(patternHash);
  key += ":";
  for (std::size_t index = 0; index < experts.size(); ++index) {
    if (index > 0U) {
      key += "-";
    }
    key += std::to_string(experts[index].expert);
  }
  return key;
}

SparsityCacheSnapshot
SparsityAwareCache::Snapshot(const bool lastLookupHit, std::string key,
                             const std::size_t patternHash,
                             const bool lastEntryWarm) const {
  SparsityCacheSnapshot snapshot;
  snapshot.entryCount = entries_.size();
  snapshot.warmEntryCount = static_cast<std::size_t>(std::count_if(
      entries_.begin(), entries_.end(),
      [](const auto &entry) { return entry.second.warm; }));
  snapshot.hitCount = hitCount_;
  snapshot.missCount = missCount_;
  const std::size_t denominator = hitCount_ + missCount_;
  snapshot.hitRatio = denominator == 0U ? 0.0
                                        : static_cast<double>(hitCount_) /
                                              static_cast<double>(denominator);
  snapshot.lastLookupHit = lastLookupHit;
  snapshot.lastEntryWarm = lastEntryWarm;
  snapshot.lastKey = std::move(key);
  snapshot.lastPatternHash = patternHash;
  return snapshot;
}

} // namespace us4
