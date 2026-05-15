#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>

namespace us4 {

struct PrefixCacheEntry {
  std::string prefix;
  std::size_t refCount = 0;
};

class PrefixCache {
 public:
  void Retain(std::string prefix);
  void Release(const std::string& prefix);
  std::optional<PrefixCacheEntry> Lookup(const std::string& prefix) const;
  std::size_t EntryCount() const;

 private:
  std::unordered_map<std::string, PrefixCacheEntry> entries_;
};

}  // namespace us4
