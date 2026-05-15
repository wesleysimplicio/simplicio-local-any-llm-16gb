#include "core/gqa_attention.h"

#include "cpu/scalar_attention.h"

namespace us4 {

bool GqaAttention(const Tensor& query,
                  const Tensor& key,
                  const Tensor& value,
                  const std::size_t queryHeads,
                  const std::size_t kvHeads,
                  Tensor& output,
                  std::string* error) {
  if (queryHeads == 0 || kvHeads == 0 || queryHeads < kvHeads || (queryHeads % kvHeads) != 0) {
    if (error != nullptr) {
      *error = "invalid GQA head relationship";
    }
    return false;
  }
  return ScalarAttention(query, key, value, output, false, {}, error);
}

}  // namespace us4
