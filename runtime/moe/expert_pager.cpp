#include "moe/expert_pager.h"

#include <algorithm>

namespace us4 {

ExpertPager::ExpertPager(const std::size_t residentLimit)
    : residentLimit_(residentLimit) {}

void ExpertPager::Touch(std::string expertId) {
  ++hits_[expertId];
  if (std::find(resident_.begin(), resident_.end(), expertId) ==
      resident_.end()) {
    ++loadCount_;
    resident_.push_back(std::move(expertId));
  } else {
    ++reuseCount_;
  }
  while (resident_.size() > residentLimit_) {
    auto victim =
        std::min_element(resident_.begin(), resident_.end(),
                         [&](const std::string &lhs, const std::string &rhs) {
                           return hits_[lhs] < hits_[rhs];
                         });
    if (victim == resident_.end()) {
      break;
    }
    ++evictionCount_;
    resident_.erase(victim);
  }
}

bool ExpertPager::IsResident(const std::string &expertId) const {
  return std::find(resident_.begin(), resident_.end(), expertId) !=
         resident_.end();
}

std::size_t ExpertPager::ResidentCount() const { return resident_.size(); }

std::size_t ExpertPager::LoadCount() const { return loadCount_; }

std::size_t ExpertPager::EvictionCount() const { return evictionCount_; }

std::size_t ExpertPager::ReuseCount() const { return reuseCount_; }

ExpertPagerSnapshot ExpertPager::Snapshot() const {
  return {
      .residentCount = resident_.size(),
      .loadCount = loadCount_,
      .evictionCount = evictionCount_,
      .reuseCount = reuseCount_,
      .residents = resident_,
  };
}

} // namespace us4
