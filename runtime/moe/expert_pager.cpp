#include "moe/expert_pager.h"

#include <algorithm>
#include <cstddef>

namespace us4 {

ExpertPager::ExpertPager(const std::size_t residentLimit,
                         const std::size_t learnedPinThreshold,
                         const std::size_t learnedPinLimit)
    : residentLimit_(residentLimit), learnedPinThreshold_(learnedPinThreshold),
      learnedPinLimit_(learnedPinLimit) {}

void ExpertPager::Touch(std::string expertId) {
  lastTouchPromotedPin_ = false;
  ++hits_[expertId];
  PromotePinIfEligible(expertId);
  if (std::find(resident_.begin(), resident_.end(), expertId) ==
      resident_.end()) {
    ++loadCount_;
    resident_.push_back(std::move(expertId));
  } else {
    ++reuseCount_;
  }
  while (resident_.size() > residentLimit_) {
    const std::size_t victimIndex = EvictionCandidateIndex();
    if (victimIndex >= resident_.size()) {
      break;
    }
    ++evictionCount_;
    resident_.erase(resident_.begin() +
                    static_cast<std::ptrdiff_t>(victimIndex));
  }
}

bool ExpertPager::IsResident(const std::string &expertId) const {
  return std::find(resident_.begin(), resident_.end(), expertId) !=
         resident_.end();
}

bool ExpertPager::IsPinned(const std::string &expertId) const {
  const auto it = pinned_.find(expertId);
  return it != pinned_.end() && it->second;
}

std::size_t ExpertPager::ResidentCount() const { return resident_.size(); }

std::size_t ExpertPager::LoadCount() const { return loadCount_; }

std::size_t ExpertPager::EvictionCount() const { return evictionCount_; }

std::size_t ExpertPager::ReuseCount() const { return reuseCount_; }

std::size_t ExpertPager::LearnedPinCount() const {
  return static_cast<std::size_t>(std::count_if(
      pinned_.begin(), pinned_.end(),
      [](const auto &entry) { return entry.second; }));
}

std::size_t ExpertPager::PinPromotionCount() const { return pinPromotionCount_; }

ExpertPagerSnapshot ExpertPager::Snapshot() const {
  std::vector<std::string> pinnedResidents;
  pinnedResidents.reserve(resident_.size());
  for (const std::string &resident : resident_) {
    if (IsPinned(resident)) {
      pinnedResidents.push_back(resident);
    }
  }
  return {
      .residentCount = resident_.size(),
      .loadCount = loadCount_,
      .evictionCount = evictionCount_,
      .reuseCount = reuseCount_,
      .learnedPinCount = LearnedPinCount(),
      .pinPromotionCount = pinPromotionCount_,
      .lastTouchPromotedPin = lastTouchPromotedPin_,
      .residents = resident_,
      .pinnedResidents = std::move(pinnedResidents),
  };
}

void ExpertPager::PromotePinIfEligible(const std::string &expertId) {
  if (learnedPinThreshold_ == 0U || learnedPinLimit_ == 0U) {
    return;
  }
  if (hits_[expertId] < learnedPinThreshold_ || IsPinned(expertId)) {
    return;
  }
  if (LearnedPinCount() >= learnedPinLimit_) {
    return;
  }
  pinned_[expertId] = true;
  ++pinPromotionCount_;
  lastTouchPromotedPin_ = true;
}

std::size_t ExpertPager::EvictionCandidateIndex() const {
  const auto chooseLeastHit =
      [&](const std::vector<std::size_t> &indices) -> std::size_t {
    if (indices.empty()) {
      return resident_.size();
    }
    return *std::min_element(
        indices.begin(), indices.end(),
        [&](const std::size_t lhsIndex, const std::size_t rhsIndex) {
          const std::string &lhs = resident_[lhsIndex];
          const std::string &rhs = resident_[rhsIndex];
          if (hits_.at(lhs) != hits_.at(rhs)) {
            return hits_.at(lhs) < hits_.at(rhs);
          }
          return lhsIndex < rhsIndex;
        });
  };

  std::vector<std::size_t> unpinnedIndices;
  std::vector<std::size_t> pinnedIndices;
  for (std::size_t index = 0; index < resident_.size(); ++index) {
    if (IsPinned(resident_[index])) {
      pinnedIndices.push_back(index);
    } else {
      unpinnedIndices.push_back(index);
    }
  }

  const std::size_t unpinnedVictim = chooseLeastHit(unpinnedIndices);
  if (unpinnedVictim < resident_.size()) {
    return unpinnedVictim;
  }
  return chooseLeastHit(pinnedIndices);
}

} // namespace us4
