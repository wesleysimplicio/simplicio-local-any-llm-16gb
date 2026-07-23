#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace us4 {

struct ExpertPagerSnapshot {
  std::size_t residentCount = 0;
  std::size_t loadCount = 0;
  std::size_t evictionCount = 0;
  std::size_t reuseCount = 0;
  std::size_t learnedPinCount = 0;
  std::size_t pinPromotionCount = 0;
  std::size_t pinDemotionCount = 0;
  std::size_t lookupCount = 0;
  double hitRatio = 0.0;
  std::size_t expertsCoveringHalfOfTouches = 0;
  bool lastTouchPromotedPin = false;
  bool lastTouchHit = false;
  std::vector<std::string> residents;
  std::vector<std::string> pinnedResidents;
};

struct ExpertUsage {
  std::string expertId;
  std::size_t touches = 0;
  bool pinned = false;
};

class ExpertPager {
public:
  explicit ExpertPager(std::size_t residentLimit = 2,
                       std::size_t learnedPinThreshold = 3,
                       std::size_t learnedPinLimit = 2,
                       std::size_t pinRebalanceInterval = 1);

  void Touch(std::string expertId);
  bool IsResident(const std::string &expertId) const;
  bool IsPinned(const std::string &expertId) const;
  std::size_t ResidentCount() const;
  std::size_t LoadCount() const;
  std::size_t EvictionCount() const;
  std::size_t ReuseCount() const;
  std::size_t LearnedPinCount() const;
  std::size_t PinPromotionCount() const;
  std::size_t PinDemotionCount() const;
  void RestoreUsageHistogram(const std::vector<ExpertUsage> &usage);
  std::vector<std::string> WarmupLearnedPins();
  std::vector<ExpertUsage> UsageHistogram() const;
  std::string ExportUsageJson() const;
  ExpertPagerSnapshot Snapshot() const;

private:
  void RebalanceLearnedPins();
  std::size_t EvictionCandidateIndex() const;
  std::size_t ExpertsCoveringHalfOfTouches() const;

  std::size_t residentLimit_ = 2;
  std::size_t learnedPinThreshold_ = 3;
  std::size_t learnedPinLimit_ = 2;
  std::size_t pinRebalanceInterval_ = 1;
  std::vector<std::string> resident_;
  std::unordered_map<std::string, std::size_t> hits_;
  std::unordered_map<std::string, bool> pinned_;
  std::size_t loadCount_ = 0;
  std::size_t evictionCount_ = 0;
  std::size_t reuseCount_ = 0;
  std::size_t pinPromotionCount_ = 0;
  std::size_t pinDemotionCount_ = 0;
  bool lastTouchPromotedPin_ = false;
  bool lastTouchHit_ = false;
};

} // namespace us4
