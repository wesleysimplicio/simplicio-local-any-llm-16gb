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
  std::vector<std::string> residents;
};

class ExpertPager {
public:
  explicit ExpertPager(std::size_t residentLimit = 2);

  void Touch(std::string expertId);
  bool IsResident(const std::string &expertId) const;
  std::size_t ResidentCount() const;
  std::size_t LoadCount() const;
  std::size_t EvictionCount() const;
  std::size_t ReuseCount() const;
  ExpertPagerSnapshot Snapshot() const;

private:
  std::size_t residentLimit_ = 2;
  std::vector<std::string> resident_;
  std::unordered_map<std::string, std::size_t> hits_;
  std::size_t loadCount_ = 0;
  std::size_t evictionCount_ = 0;
  std::size_t reuseCount_ = 0;
};

} // namespace us4
