#include "moe/expert_pager.h"

#include <algorithm>
#include <cstddef>
#include <sstream>

namespace us4 {

namespace {

std::string EscapeJson(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char ch : value) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(ch);
      break;
    }
  }
  return escaped;
}

} // namespace

ExpertPager::ExpertPager(const std::size_t residentLimit,
                         const std::size_t learnedPinThreshold,
                         const std::size_t learnedPinLimit,
                         const std::size_t pinRebalanceInterval)
    : residentLimit_(residentLimit), learnedPinThreshold_(learnedPinThreshold),
      learnedPinLimit_(learnedPinLimit),
      pinRebalanceInterval_(
          std::max<std::size_t>(1U, pinRebalanceInterval)) {}

void ExpertPager::Touch(std::string expertId) {
  lastTouchPromotedPin_ = false;
  lastTouchHit_ = IsResident(expertId);
  ++hits_[expertId];
  const std::size_t lookupCount = loadCount_ + reuseCount_ + 1U;
  const bool canFillPinBudget = LearnedPinCount() < learnedPinLimit_;
  if ((canFillPinBudget && hits_[expertId] == learnedPinThreshold_) ||
      (lookupCount % pinRebalanceInterval_) == 0U) {
    RebalanceLearnedPins();
  }
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

std::size_t ExpertPager::PinDemotionCount() const { return pinDemotionCount_; }

void ExpertPager::RestoreUsageHistogram(
    const std::vector<ExpertUsage> &usage) {
  hits_.clear();
  pinned_.clear();
  for (const ExpertUsage &entry : usage) {
    if (!entry.expertId.empty() && entry.touches > 0U) {
      hits_[entry.expertId] += entry.touches;
    }
  }
  RebalanceLearnedPins();
  pinPromotionCount_ = 0U;
  pinDemotionCount_ = 0U;
  lastTouchPromotedPin_ = false;
}

std::vector<std::string> ExpertPager::WarmupLearnedPins() {
  std::vector<std::string> loaded;
  for (const ExpertUsage &usage : UsageHistogram()) {
    if (!usage.pinned || IsResident(usage.expertId) ||
        resident_.size() >= residentLimit_) {
      continue;
    }
    resident_.push_back(usage.expertId);
    loaded.push_back(usage.expertId);
    ++loadCount_;
  }
  return loaded;
}

std::vector<ExpertUsage> ExpertPager::UsageHistogram() const {
  std::vector<ExpertUsage> usage;
  usage.reserve(hits_.size());
  for (const auto &[expertId, touches] : hits_) {
    usage.push_back(
        {.expertId = expertId, .touches = touches, .pinned = IsPinned(expertId)});
  }
  std::sort(usage.begin(), usage.end(),
            [](const ExpertUsage &lhs, const ExpertUsage &rhs) {
              if (lhs.touches != rhs.touches) {
                return lhs.touches > rhs.touches;
              }
              return lhs.expertId < rhs.expertId;
            });
  return usage;
}

std::string ExpertPager::ExportUsageJson() const {
  std::ostringstream stream;
  stream << "{\"version\":1,\"lookups\":" << (loadCount_ + reuseCount_)
         << ",\"hits\":" << reuseCount_ << ",\"misses\":" << loadCount_
         << ",\"experts_covering_50_percent\":"
         << ExpertsCoveringHalfOfTouches() << ",\"experts\":[";
  const std::vector<ExpertUsage> histogram = UsageHistogram();
  for (std::size_t index = 0; index < histogram.size(); ++index) {
    if (index > 0U) {
      stream << ',';
    }
    stream << "{\"id\":\"" << EscapeJson(histogram[index].expertId)
           << "\",\"touches\":" << histogram[index].touches
           << ",\"pinned\":" << (histogram[index].pinned ? "true" : "false")
           << '}';
  }
  stream << "]}";
  return stream.str();
}

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
      .pinDemotionCount = pinDemotionCount_,
      .lookupCount = loadCount_ + reuseCount_,
      .hitRatio = (loadCount_ + reuseCount_) == 0U
                      ? 0.0
                      : static_cast<double>(reuseCount_) /
                            static_cast<double>(loadCount_ + reuseCount_),
      .expertsCoveringHalfOfTouches = ExpertsCoveringHalfOfTouches(),
      .lastTouchPromotedPin = lastTouchPromotedPin_,
      .lastTouchHit = lastTouchHit_,
      .residents = resident_,
      .pinnedResidents = std::move(pinnedResidents),
  };
}

void ExpertPager::RebalanceLearnedPins() {
  if (learnedPinThreshold_ == 0U || learnedPinLimit_ == 0U) {
    return;
  }

  std::vector<ExpertUsage> candidates = UsageHistogram();
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [this](const ExpertUsage &usage) {
                       return usage.touches < learnedPinThreshold_;
                     }),
      candidates.end());
  if (candidates.size() > learnedPinLimit_) {
    candidates.resize(learnedPinLimit_);
  }

  std::unordered_map<std::string, bool> nextPinned;
  for (const ExpertUsage &candidate : candidates) {
    nextPinned[candidate.expertId] = true;
    if (!IsPinned(candidate.expertId)) {
      ++pinPromotionCount_;
      lastTouchPromotedPin_ = true;
    }
  }
  for (const auto &[expertId, wasPinned] : pinned_) {
    if (wasPinned && !nextPinned[expertId]) {
      ++pinDemotionCount_;
    }
  }
  pinned_ = std::move(nextPinned);
}

std::size_t ExpertPager::ExpertsCoveringHalfOfTouches() const {
  const std::vector<ExpertUsage> histogram = UsageHistogram();
  std::size_t total = 0U;
  for (const ExpertUsage &usage : histogram) {
    total += usage.touches;
  }
  if (total == 0U) {
    return 0U;
  }
  const std::size_t half = (total / 2U) + (total % 2U);
  std::size_t cumulative = 0U;
  for (std::size_t index = 0; index < histogram.size(); ++index) {
    cumulative += histogram[index].touches;
    if (cumulative >= half) {
      return index + 1U;
    }
  }
  return histogram.size();
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
