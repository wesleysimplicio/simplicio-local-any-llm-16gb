#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace us4 {

class ExpertPager {
 public:
  explicit ExpertPager(std::size_t residentLimit = 2);

  void Touch(std::string expertId);
  bool IsResident(const std::string& expertId) const;
  std::size_t ResidentCount() const;

 private:
  std::size_t residentLimit_ = 2;
  std::vector<std::string> resident_;
  std::unordered_map<std::string, std::size_t> hits_;
};

}  // namespace us4
