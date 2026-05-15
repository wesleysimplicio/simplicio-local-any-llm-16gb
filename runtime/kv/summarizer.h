#pragma once

#include <vector>

namespace us4 {

class Summarizer {
 public:
  std::vector<float> Summarize(const std::vector<float>& values) const;
};

}  // namespace us4
