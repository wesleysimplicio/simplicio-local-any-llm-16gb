#pragma once

#include <string>

#include "core/tensor.h"

namespace us4 {

bool ScalarMatmul(const Tensor& lhs, const Tensor& rhs, Tensor& output, std::string* error = nullptr);

}  // namespace us4
