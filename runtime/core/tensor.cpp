#include "core/tensor.h"

#include <algorithm>
#include <utility>

namespace us4 {

std::string_view ToString(const DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "fp32";
    case DType::kFloat16:
      return "fp16";
    case DType::kBFloat16:
      return "bf16";
    case DType::kInt8:
      return "int8";
    case DType::kInt4:
      return "int4";
  }
  return "unknown";
}

std::string_view ToString(const DeviceType device) {
  switch (device) {
    case DeviceType::kCpu:
      return "cpu";
    case DeviceType::kMlx:
      return "mlx";
    case DeviceType::kMetal:
      return "metal";
    case DeviceType::kNeon:
      return "neon";
    case DeviceType::kAne:
      return "ane";
    case DeviceType::kUnknown:
      return "unknown";
  }
  return "unknown";
}

std::size_t DTypeBitWidth(const DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return 32;
    case DType::kFloat16:
    case DType::kBFloat16:
      return 16;
    case DType::kInt8:
      return 8;
    case DType::kInt4:
      return 4;
  }
  return 0;
}

Tensor::Tensor(std::vector<std::size_t> shape, const DType dtype, const DeviceType device)
    : shape_(std::move(shape)),
      strides_(ComputeContiguousStrides(shape_)),
      dtype_(dtype),
      device_(device),
      storage_(ByteSize()) {}

const std::vector<std::size_t>& Tensor::Shape() const { return shape_; }

const std::vector<std::size_t>& Tensor::Strides() const { return strides_; }

DType Tensor::dtype() const { return dtype_; }

DeviceType Tensor::device() const { return device_; }

std::size_t Tensor::Rank() const { return shape_.size(); }

std::size_t Tensor::ElementCount() const { return ComputeElementCount(shape_); }

std::size_t Tensor::ByteSize() const {
  const std::size_t bits = ElementCount() * DTypeBitWidth(dtype_);
  return bits == 0 ? 0 : (bits + 7U) / 8U;
}

bool Tensor::Empty() const { return ElementCount() == 0; }

bool Tensor::IsContiguous() const { return strides_ == ComputeContiguousStrides(shape_); }

const std::byte* Tensor::Data() const { return storage_.data(); }

std::byte* Tensor::MutableData() { return storage_.data(); }

const float* Tensor::DataAsFloat32() const {
  return dtype_ == DType::kFloat32 ? reinterpret_cast<const float*>(storage_.data()) : nullptr;
}

float* Tensor::MutableDataAsFloat32() {
  return dtype_ == DType::kFloat32 ? reinterpret_cast<float*>(storage_.data()) : nullptr;
}

bool Tensor::Reshape(std::vector<std::size_t> shape) {
  if (ComputeElementCount(shape) != ElementCount()) {
    return false;
  }

  shape_ = std::move(shape);
  strides_ = ComputeContiguousStrides(shape_);
  return true;
}

void Tensor::FillZero() { std::fill(storage_.begin(), storage_.end(), std::byte{0}); }

std::vector<std::size_t> Tensor::ComputeContiguousStrides(const std::vector<std::size_t>& shape) {
  std::vector<std::size_t> strides(shape.size(), 1);
  for (std::size_t index = shape.size(); index > 0; --index) {
    if (index < shape.size()) {
      strides[index - 1] = strides[index] * shape[index];
    }
  }
  return strides;
}

std::size_t Tensor::ComputeElementCount(const std::vector<std::size_t>& shape) {
  if (shape.empty()) {
    return 0;
  }

  std::size_t count = 1;
  for (const std::size_t dim : shape) {
    count *= dim;
  }
  return count;
}

}  // namespace us4
