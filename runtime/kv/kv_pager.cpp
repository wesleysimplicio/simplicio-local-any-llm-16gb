#include "kv/kv_pager.h"

#include <algorithm>

namespace us4 {

namespace {

bool IsRowLayoutValid(const std::vector<float> &keys,
                      const std::vector<float> &values,
                      const std::size_t rowWidth) {
  return rowWidth != 0 && !keys.empty() && keys.size() == values.size() &&
         (keys.size() % rowWidth) == 0;
}

} // namespace

KvPager::KvPager(const std::size_t hotLimit) : hotLimit_(hotLimit) {}

void KvPager::Append(std::string key, std::vector<float> values) {
  std::vector<float> mirroredValues = values;
  const std::size_t rowWidth = mirroredValues.size();
  Append(std::move(key), std::move(mirroredValues), std::move(values),
         rowWidth);
}

void KvPager::Append(std::string key, std::vector<float> keys,
                     std::vector<float> values, const std::size_t rowWidth) {
  const auto existing = index_.find(key);
  if (existing != index_.end()) {
    KvPage &page = pages_[existing->second];
    page.keys = std::move(keys);
    page.values = std::move(values);
    page.rowWidth = rowWidth;
    page.rowCount = IsRowLayoutValid(page.keys, page.values, rowWidth)
                        ? page.keys.size() / rowWidth
                        : 0U;
    Promote(page);
    EvictIfNeeded();
    return;
  }

  KvPage page;
  page.key = std::move(key);
  page.keys = std::move(keys);
  page.values = std::move(values);
  page.rowWidth = rowWidth;
  page.rowCount = IsRowLayoutValid(page.keys, page.values, rowWidth)
                      ? page.keys.size() / rowWidth
                      : 0U;
  Stamp(page);
  index_[page.key] = pages_.size();
  pages_.push_back(std::move(page));
  EvictIfNeeded();
}

bool KvPager::AppendRow(const std::string &key,
                        const std::vector<float> &keyRow,
                        const std::vector<float> &valueRow) {
  const auto it = index_.find(key);
  if (it == index_.end()) {
    return false;
  }

  KvPage &page = pages_[it->second];
  if (page.rowWidth == 0 || keyRow.size() != page.rowWidth ||
      valueRow.size() != page.rowWidth) {
    return false;
  }

  page.keys.insert(page.keys.end(), keyRow.begin(), keyRow.end());
  page.values.insert(page.values.end(), valueRow.begin(), valueRow.end());
  ++page.rowCount;
  Promote(page);
  EvictIfNeeded();
  return true;
}

std::optional<KvPage> KvPager::Lookup(const std::string &key) {
  const auto it = index_.find(key);
  if (it == index_.end()) {
    return std::nullopt;
  }
  Touch(key);
  return pages_[it->second];
}

void KvPager::Touch(const std::string &key) {
  const auto it = index_.find(key);
  if (it == index_.end()) {
    return;
  }
  Promote(pages_[it->second]);
  EvictIfNeeded();
}

void KvPager::EvictIfNeeded() {
  while (HotPageCount() > hotLimit_) {
    auto victim = pages_.end();
    for (auto it = pages_.begin(); it != pages_.end(); ++it) {
      if (it->tier != KvTier::kHot) {
        continue;
      }
      if (victim == pages_.end() || it->hitCount < victim->hitCount ||
          (it->hitCount == victim->hitCount &&
           it->lastTouch < victim->lastTouch)) {
        victim = it;
      }
    }
    if (victim == pages_.end()) {
      break;
    }
    victim->tier = KvTier::kWarm;
  }

  while (WarmPageCount() > hotLimit_) {
    auto victim = pages_.end();
    for (auto it = pages_.begin(); it != pages_.end(); ++it) {
      if (it->tier != KvTier::kWarm) {
        continue;
      }
      if (victim == pages_.end() || it->hitCount < victim->hitCount ||
          (it->hitCount == victim->hitCount &&
           it->lastTouch < victim->lastTouch)) {
        victim = it;
      }
    }
    if (victim == pages_.end()) {
      break;
    }
    victim->tier = KvTier::kCold;
  }
}

std::size_t KvPager::PageCount() const { return pages_.size(); }

std::size_t KvPager::HotPageCount() const {
  return CountPagesWithTier(KvTier::kHot);
}

std::size_t KvPager::WarmPageCount() const {
  return CountPagesWithTier(KvTier::kWarm);
}

std::size_t KvPager::ColdPageCount() const {
  return CountPagesWithTier(KvTier::kCold);
}

void KvPager::Promote(KvPage &page) {
  ++page.hitCount;
  page.tier = KvTier::kHot;
  Stamp(page);
}

void KvPager::Stamp(KvPage &page) { page.lastTouch = ++accessClock_; }

std::size_t KvPager::CountPagesWithTier(const KvTier tier) const {
  std::size_t count = 0;
  for (const KvPage &page : pages_) {
    if (page.tier == tier) {
      ++count;
    }
  }
  return count;
}

} // namespace us4
