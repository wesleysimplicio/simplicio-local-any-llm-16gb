#include "moe/router.h"

#include <algorithm>

namespace us4 {

std::vector<ExpertScore> Router::TopK(const std::vector<float>& logits, const std::size_t k) const {
  std::vector<ExpertScore> scores;
  scores.reserve(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    scores.push_back({index, logits[index]});
  }
  std::sort(scores.begin(), scores.end(), [](const ExpertScore& lhs, const ExpertScore& rhs) {
    return lhs.score > rhs.score;
  });
  if (scores.size() > k) {
    scores.resize(k);
  }
  return scores;
}

}  // namespace us4
