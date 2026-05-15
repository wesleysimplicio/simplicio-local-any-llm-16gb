#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace us4 {

enum class KvTier {
  kHot,
  kWarm,
  kCold,
};

struct KvPage {
  std::string key;
  std::vector<float> values;
  KvTier tier = KvTier::kHot;
  std::size_t hitCount = 0;
};

class KvPager {
 public:
  explicit KvPager(std::size_t hotLimit = 4);

  void Append(std::string key, std::vector<float> values);
  std::optional<KvPage> Lookup(const std::string& key);
  void Touch(const std::string& key);
  void EvictIfNeeded();
  std::size_t PageCount() const;
  std::size_t HotPageCount() const;

 private:
  std::size_t hotLimit_ = 4;
  std::vector<KvPage> pages_;
  std::unordered_map<std::string, std::size_t> index_;
};

}  // namespace us4
