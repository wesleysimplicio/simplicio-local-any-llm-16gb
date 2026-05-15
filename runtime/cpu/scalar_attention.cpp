#include "cpu/scalar_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace us4 {

namespace {

bool WriteError(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool ValidateRank2Float(const Tensor& tensor, std::string* error, const char* message) {
  if (tensor.dtype() != DType::kFloat32 || tensor.DataAsFloat32() == nullptr) {
    return WriteError(error, message);
  }
  if (tensor.Rank() != 2 || !tensor.IsContiguous()) {
    return WriteError(error, message);
  }
  return true;
}

void CopyRows(const Tensor& source, std::vector<float>& target, std::size_t targetOffsetRows, std::size_t width) {
  const float* data = source.DataAsFloat32();
  const std::size_t rows = source.Shape()[0];
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t sourceOffset = row * width;
    const std::size_t targetOffset = (targetOffsetRows + row) * width;
    std::copy_n(data + sourceOffset, width, target.begin() + static_cast<std::ptrdiff_t>(targetOffset));
  }
}

}  // namespace

bool ScalarAttention(const Tensor& query,
                     const Tensor& key,
                     const Tensor& value,
                     Tensor& output,
                     const bool causalMask,
                     const AttentionCacheView cache,
                     std::string* error) {
  if (!ValidateRank2Float(query, error, "query must be contiguous rank-2 fp32") ||
      !ValidateRank2Float(key, error, "key must be contiguous rank-2 fp32") ||
      !ValidateRank2Float(value, error, "value must be contiguous rank-2 fp32")) {
    return false;
  }

  if (output.dtype() != DType::kFloat32 || output.MutableDataAsFloat32() == nullptr || output.Rank() != 2 ||
      !output.IsContiguous()) {
    return WriteError(error, "output must be writable rank-2 fp32");
  }

  if (query.Shape()[1] != key.Shape()[1]) {
    return WriteError(error, "query and key hidden sizes must match");
  }
  if (key.Shape()[0] != value.Shape()[0]) {
    return WriteError(error, "key and value token counts must match");
  }
  if (output.Shape()[0] != query.Shape()[0] || output.Shape()[1] != value.Shape()[1]) {
    return WriteError(error, "output shape does not match attention result");
  }

  const std::size_t cacheRows = cache.keys == nullptr ? 0 : cache.keys->Shape()[0];
  if ((cache.keys == nullptr) != (cache.values == nullptr)) {
    return WriteError(error, "cache keys and values must be provided together");
  }
  if (cache.keys != nullptr) {
    if (!ValidateRank2Float(*cache.keys, error, "cache keys must be contiguous rank-2 fp32") ||
        !ValidateRank2Float(*cache.values, error, "cache values must be contiguous rank-2 fp32")) {
      return false;
    }
    if (cache.keys->Shape()[1] != key.Shape()[1] || cache.values->Shape()[1] != value.Shape()[1] ||
        cache.keys->Shape()[0] != cache.values->Shape()[0]) {
      return WriteError(error, "cache tensor shapes must match active kv tensors");
    }
  }

  const std::size_t queryRows = query.Shape()[0];
  const std::size_t hidden = query.Shape()[1];
  const std::size_t activeRows = key.Shape()[0];
  const std::size_t valueWidth = value.Shape()[1];
  const std::size_t totalRows = cacheRows + activeRows;
  const float scale = 1.0F / std::sqrt(static_cast<float>(hidden));

  std::vector<float> mergedKeys(totalRows * hidden, 0.0F);
  std::vector<float> mergedValues(totalRows * valueWidth, 0.0F);
  if (cache.keys != nullptr) {
    CopyRows(*cache.keys, mergedKeys, 0, hidden);
    CopyRows(*cache.values, mergedValues, 0, valueWidth);
  }
  CopyRows(key, mergedKeys, cacheRows, hidden);
  CopyRows(value, mergedValues, cacheRows, valueWidth);

  const float* queryData = query.DataAsFloat32();
  float* outputData = output.MutableDataAsFloat32();

  for (std::size_t row = 0; row < queryRows; ++row) {
    std::vector<float> scores(totalRows, 0.0F);
    const std::size_t visibleRows = causalMask ? (cacheRows + row + 1) : totalRows;
    if (visibleRows == 0) {
      return WriteError(error, "attention requires at least one visible kv row");
    }

    for (std::size_t kvRow = 0; kvRow < visibleRows; ++kvRow) {
      float dot = 0.0F;
      for (std::size_t col = 0; col < hidden; ++col) {
        dot += queryData[row * hidden + col] * mergedKeys[kvRow * hidden + col];
      }
      scores[kvRow] = dot * scale;
    }

    const float maxScore = *std::max_element(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(visibleRows));
    float denominator = 0.0F;
    for (std::size_t kvRow = 0; kvRow < visibleRows; ++kvRow) {
      scores[kvRow] = std::exp(scores[kvRow] - maxScore);
      denominator += scores[kvRow];
    }
    if (denominator <= std::numeric_limits<float>::epsilon()) {
      return WriteError(error, "attention softmax denominator is too small");
    }

    for (std::size_t outCol = 0; outCol < valueWidth; ++outCol) {
      float weightedSum = 0.0F;
      for (std::size_t kvRow = 0; kvRow < visibleRows; ++kvRow) {
        weightedSum += (scores[kvRow] / denominator) * mergedValues[kvRow * valueWidth + outCol];
      }
      outputData[row * valueWidth + outCol] = weightedSum;
    }
  }

  return true;
}

}  // namespace us4
