#pragma once

#include <string>

#include "cpu/scalar_attention.h"

namespace us4 {

bool NeonAttention(const Tensor& query,
                   const Tensor& key,
                   const Tensor& value,
                   Tensor& output,
                   bool causalMask,
                   AttentionCacheView cache = {},
                   std::string* error = nullptr);

}  // namespace us4
