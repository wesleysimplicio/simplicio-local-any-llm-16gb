#pragma once

#include <cstddef>
#include <vector>

namespace us4 {

struct ExpertScore {
  std::size_t expert = 0;
  float score = 0.0F;
};

class Router {
 public:
  std::vector<ExpertScore> TopK(const std::vector<float>& logits, std::size_t k) const;
};

}  // namespace us4
