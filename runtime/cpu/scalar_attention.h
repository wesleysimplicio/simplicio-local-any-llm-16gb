#pragma once

#include <string>

#include "core/tensor.h"

namespace us4 {

struct AttentionCacheView {
  const Tensor* keys = nullptr;
  const Tensor* values = nullptr;
};

bool ScalarAttention(const Tensor& query,
                     const Tensor& key,
                     const Tensor& value,
                     Tensor& output,
                     bool causalMask,
                     AttentionCacheView cache = {},
                     std::string* error = nullptr);

}  // namespace us4
