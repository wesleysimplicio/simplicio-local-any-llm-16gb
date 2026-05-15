#include "kv/kv_pager.h"

#include <algorithm>

namespace us4 {

KvPager::KvPager(const std::size_t hotLimit) : hotLimit_(hotLimit) {}

void KvPager::Append(std::string key, std::vector<float> values) {
  const auto existing = index_.find(key);
  if (existing != index_.end()) {
    KvPage& page = pages_[existing->second];
    page.values = std::move(values);
    page.tier = KvTier::kHot;
    ++page.hitCount;
    EvictIfNeeded();
    return;
  }

  KvPage page;
  page.key = std::move(key);
  page.values = std::move(values);
  index_[page.key] = pages_.size();
  pages_.push_back(std::move(page));
  EvictIfNeeded();
}

std::optional<KvPage> KvPager::Lookup(const std::string& key) {
  const auto it = index_.find(key);
  if (it == index_.end()) {
    return std::nullopt;
  }
  Touch(key);
  return pages_[it->second];
}

void KvPager::Touch(const std::string& key) {
  const auto it = index_.find(key);
  if (it == index_.end()) {
    return;
  }
  KvPage& page = pages_[it->second];
  ++page.hitCount;
  page.tier = KvTier::kHot;
  EvictIfNeeded();
}

void KvPager::EvictIfNeeded() {
  while (HotPageCount() > hotLimit_) {
    auto victim = std::min_element(
        pages_.begin(), pages_.end(),
        [](const KvPage& lhs, const KvPage& rhs) { return lhs.hitCount < rhs.hitCount; });
    if (victim == pages_.end()) {
      break;
    }
    victim->tier = victim->tier == KvTier::kWarm ? KvTier::kCold : KvTier::kWarm;
  }
}

std::size_t KvPager::PageCount() const { return pages_.size(); }

std::size_t KvPager::HotPageCount() const {
  std::size_t count = 0;
  for (const KvPage& page : pages_) {
    if (page.tier == KvTier::kHot) {
      ++count;
    }
  }
  return count;
}

}  // namespace us4
