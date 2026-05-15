#pragma once

#include <cstddef>
#include <string>

#include "core/tensor.h"

namespace us4 {

bool GqaAttention(const Tensor& query,
                  const Tensor& key,
                  const Tensor& value,
                  std::size_t queryHeads,
                  std::size_t kvHeads,
                  Tensor& output,
                  std::string* error = nullptr);

}  // namespace us4
