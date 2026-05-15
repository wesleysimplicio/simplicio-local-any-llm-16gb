#include "kv/summarizer.h"

namespace us4 {

std::vector<float> Summarizer::Summarize(const std::vector<float>& values) const {
  if (values.empty()) {
    return {};
  }
  float sum = 0.0F;
  for (const float value : values) {
    sum += value;
  }
  return {sum / static_cast<float>(values.size())};
}

}  // namespace us4
