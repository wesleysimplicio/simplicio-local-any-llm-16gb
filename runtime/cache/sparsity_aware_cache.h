#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "moe/router.h"

namespace us4 {

struct SparsityCacheEntry {
  std::string family;
  std::string key;
  std::size_t patternHash = 0;
  std::vector<std::size_t> experts;
  std::size_t uses = 0;
  std::size_t hits = 0;
  bool warm = false;
};

struct SparsityCacheSnapshot {
  std::size_t entryCount = 0;
  std::size_t warmEntryCount = 0;
  std::size_t hitCount = 0;
  std::size_t missCount = 0;
  double hitRatio = 0.0;
  bool lastLookupHit = false;
  bool lastEntryWarm = false;
  std::string lastKey;
  std::size_t lastPatternHash = 0;
};

class SparsityAwareCache {
public:
  std::optional<SparsityCacheEntry> Lookup(std::string_view family,
                                           const RouterDecision &routing) const;
  SparsityCacheSnapshot Touch(std::string_view family,
                              const RouterDecision &routing);
  std::size_t EntryCount() const;

private:
  static std::size_t
  ComputePatternHash(const std::vector<ExpertScore> &experts);
  static std::string BuildKey(std::string_view family, std::size_t patternHash,
                              const std::vector<ExpertScore> &experts);
  SparsityCacheSnapshot Snapshot(bool lastLookupHit, std::string key,
                                 std::size_t patternHash,
                                 bool lastEntryWarm) const;

  std::unordered_map<std::string, SparsityCacheEntry> entries_;
  std::size_t hitCount_ = 0;
  std::size_t missCount_ = 0;
};

} // namespace us4
