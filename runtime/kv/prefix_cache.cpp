#include "kv/prefix_cache.h"

namespace us4 {

void PrefixCache::Retain(std::string prefix) {
  PrefixCacheEntry& entry = entries_[prefix];
  entry.prefix = std::move(prefix);
  ++entry.refCount;
}

void PrefixCache::Release(const std::string& prefix) {
  const auto it = entries_.find(prefix);
  if (it == entries_.end()) {
    return;
  }
  if (it->second.refCount > 1) {
    --it->second.refCount;
    return;
  }
  entries_.erase(it);
}

std::optional<PrefixCacheEntry> PrefixCache::Lookup(const std::string& prefix) const {
  const auto it = entries_.find(prefix);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::size_t PrefixCache::EntryCount() const { return entries_.size(); }

}  // namespace us4
