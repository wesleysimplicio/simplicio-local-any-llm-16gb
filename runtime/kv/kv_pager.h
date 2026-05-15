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
  std::vector<float> keys;
  std::vector<float> values;
  std::size_t rowWidth = 0;
  std::size_t rowCount = 0;
  KvTier tier = KvTier::kHot;
  std::size_t hitCount = 0;
  std::size_t lastTouch = 0;
};

class KvPager {
public:
  explicit KvPager(std::size_t hotLimit = 4);

  void Append(std::string key, std::vector<float> values);
  void Append(std::string key, std::vector<float> keys,
              std::vector<float> values, std::size_t rowWidth);
  bool AppendRow(const std::string &key, const std::vector<float> &keyRow,
                 const std::vector<float> &valueRow);
  std::optional<KvPage> Lookup(const std::string &key);
  void Touch(const std::string &key);
  void EvictIfNeeded();
  std::size_t PageCount() const;
  std::size_t HotPageCount() const;
  std::size_t WarmPageCount() const;
  std::size_t ColdPageCount() const;

private:
  void Promote(KvPage &page);
  void Stamp(KvPage &page);
  std::size_t CountPagesWithTier(KvTier tier) const;

  std::size_t hotLimit_ = 4;
  std::size_t accessClock_ = 0;
  std::vector<KvPage> pages_;
  std::unordered_map<std::string, std::size_t> index_;
};

} // namespace us4
