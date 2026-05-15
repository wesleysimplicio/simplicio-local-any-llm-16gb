#include "cpu/scalar_matmul.h"

namespace us4 {

namespace {

bool WriteError(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool ValidateMatrix(const Tensor& tensor, std::string* error) {
  if (tensor.dtype() != DType::kFloat32) {
    return WriteError(error, "tensor must be fp32");
  }
  if (tensor.Rank() != 2) {
    return WriteError(error, "tensor must be rank-2");
  }
  if (!tensor.IsContiguous()) {
    return WriteError(error, "tensor must be contiguous");
  }
  if (tensor.DataAsFloat32() == nullptr) {
    return WriteError(error, "tensor storage is unavailable");
  }
  return true;
}

}  // namespace

bool ScalarMatmul(const Tensor& lhs, const Tensor& rhs, Tensor& output, std::string* error) {
  if (!ValidateMatrix(lhs, error) || !ValidateMatrix(rhs, error)) {
    return false;
  }

  if (output.dtype() != DType::kFloat32 || output.Rank() != 2 || output.MutableDataAsFloat32() == nullptr ||
      !output.IsContiguous()) {
    return WriteError(error, "output must be a writable fp32 rank-2 tensor");
  }

  const std::size_t lhs_rows = lhs.Shape()[0];
  const std::size_t lhs_cols = lhs.Shape()[1];
  const std::size_t rhs_rows = rhs.Shape()[0];
  const std::size_t rhs_cols = rhs.Shape()[1];

  if (lhs_cols != rhs_rows) {
    return WriteError(error, "lhs columns must match rhs rows");
  }
  if (output.Shape()[0] != lhs_rows || output.Shape()[1] != rhs_cols) {
    return WriteError(error, "output shape does not match matmul result");
  }

  const float* lhs_data = lhs.DataAsFloat32();
  const float* rhs_data = rhs.DataAsFloat32();
  float* out_data = output.MutableDataAsFloat32();

  for (std::size_t row = 0; row < lhs_rows; ++row) {
    for (std::size_t col = 0; col < rhs_cols; ++col) {
      float accumulator = 0.0F;
      for (std::size_t inner = 0; inner < lhs_cols; ++inner) {
        accumulator += lhs_data[row * lhs_cols + inner] * rhs_data[inner * rhs_cols + col];
      }
      out_data[row * rhs_cols + col] = accumulator;
    }
  }

  return true;
}

}  // namespace us4
