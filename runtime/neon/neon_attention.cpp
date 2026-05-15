#include "neon/neon_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "cpu/scalar_attention.h"

namespace us4 {

namespace {

bool WriteError(std::string *error, const char *message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool ValidateRank2Float(const Tensor &tensor, std::string *error,
                        const char *message) {
  if (tensor.dtype() != DType::kFloat32 || tensor.DataAsFloat32() == nullptr) {
    return WriteError(error, message);
  }
  if (tensor.Rank() != 2 || !tensor.IsContiguous()) {
    return WriteError(error, message);
  }
  return true;
}

void CopyRows(const Tensor &source, std::vector<float> &target,
              const std::size_t targetOffsetRows, const std::size_t width) {
  const float *data = source.DataAsFloat32();
  const std::size_t rows = source.Shape()[0];
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t sourceOffset = row * width;
    const std::size_t targetOffset = (targetOffsetRows + row) * width;
    std::copy_n(data + sourceOffset, width,
                target.begin() + static_cast<std::ptrdiff_t>(targetOffset));
  }
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
float NeonDotProductLane4(const float *lhs, const float *rhs,
                          const std::size_t length) {
  float32x4_t accumulator = vdupq_n_f32(0.0F);
  std::size_t index = 0;
  for (; index + 4U <= length; index += 4U) {
    const float32x4_t lhsVector = vld1q_f32(lhs + index);
    const float32x4_t rhsVector = vld1q_f32(rhs + index);
    accumulator = vmlaq_f32(accumulator, lhsVector, rhsVector);
  }

  alignas(16) float lanes[4];
  vst1q_f32(lanes, accumulator);
  float result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
  for (; index < length; ++index) {
    result += lhs[index] * rhs[index];
  }
  return result;
}
#endif

} // namespace

bool NeonAttention(const Tensor &query, const Tensor &key, const Tensor &value,
                   Tensor &output, const bool causalMask,
                   const AttentionCacheView cache, std::string *error) {
#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
  return ScalarAttention(query, key, value, output, causalMask, cache, error);
#else
  if (!ValidateRank2Float(query, error,
                          "query must be contiguous rank-2 fp32") ||
      !ValidateRank2Float(key, error, "key must be contiguous rank-2 fp32") ||
      !ValidateRank2Float(value, error,
                          "value must be contiguous rank-2 fp32")) {
    return false;
  }

  if (output.dtype() != DType::kFloat32 ||
      output.MutableDataAsFloat32() == nullptr || output.Rank() != 2 ||
      !output.IsContiguous()) {
    return WriteError(error, "output must be writable rank-2 fp32");
  }

  if (query.Shape()[1] != key.Shape()[1]) {
    return WriteError(error, "query and key hidden sizes must match");
  }
  if (key.Shape()[0] != value.Shape()[0]) {
    return WriteError(error, "key and value token counts must match");
  }
  if (output.Shape()[0] != query.Shape()[0] ||
      output.Shape()[1] != value.Shape()[1]) {
    return WriteError(error, "output shape does not match attention result");
  }

  const std::size_t cacheRows =
      cache.keys == nullptr ? 0 : cache.keys->Shape()[0];
  if ((cache.keys == nullptr) != (cache.values == nullptr)) {
    return WriteError(error, "cache keys and values must be provided together");
  }
  if (cache.keys != nullptr) {
    if (!ValidateRank2Float(*cache.keys, error,
                            "cache keys must be contiguous rank-2 fp32") ||
        !ValidateRank2Float(*cache.values, error,
                            "cache values must be contiguous rank-2 fp32")) {
      return false;
    }
    if (cache.keys->Shape()[1] != key.Shape()[1] ||
        cache.values->Shape()[1] != value.Shape()[1] ||
        cache.keys->Shape()[0] != cache.values->Shape()[0]) {
      return WriteError(error,
                        "cache tensor shapes must match active kv tensors");
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

  const float *queryData = query.DataAsFloat32();
  float *outputData = output.MutableDataAsFloat32();

  for (std::size_t row = 0; row < queryRows; ++row) {
    std::vector<float> scores(totalRows, 0.0F);
    const std::size_t visibleRows =
        causalMask ? (cacheRows + row + 1U) : totalRows;
    if (visibleRows == 0U) {
      return WriteError(error,
                        "attention requires at least one visible kv row");
    }

    const float *queryRow = queryData + (row * hidden);
    for (std::size_t kvRow = 0; kvRow < visibleRows; ++kvRow) {
      const float *keyRow = mergedKeys.data() + (kvRow * hidden);
      scores[kvRow] = NeonDotProductLane4(queryRow, keyRow, hidden) * scale;
    }

    const float maxScore = *std::max_element(
        scores.begin(),
        scores.begin() + static_cast<std::ptrdiff_t>(visibleRows));
    float denominator = 0.0F;
    for (std::size_t kvRow = 0; kvRow < visibleRows; ++kvRow) {
      scores[kvRow] = std::exp(scores[kvRow] - maxScore);
      denominator += scores[kvRow];
    }
    if (denominator <= std::numeric_limits<float>::epsilon()) {
      return WriteError(error, "attention softmax denominator is too small");
    }

    for (std::size_t kvRow = 0; kvRow < visibleRows; ++kvRow) {
      scores[kvRow] /= denominator;
    }

    std::size_t outCol = 0;
    for (; outCol + 4U <= valueWidth; outCol += 4U) {
      float32x4_t weightedVector = vdupq_n_f32(0.0F);
      for (std::size_t kvRow = 0; kvRow < visibleRows; ++kvRow) {
        const float32x4_t valueVector =
            vld1q_f32(mergedValues.data() + (kvRow * valueWidth + outCol));
        weightedVector =
            vmlaq_n_f32(weightedVector, valueVector, scores[kvRow]);
      }
      vst1q_f32(outputData + (row * valueWidth + outCol), weightedVector);
    }

    for (; outCol < valueWidth; ++outCol) {
      float weightedSum = 0.0F;
      for (std::size_t kvRow = 0; kvRow < visibleRows; ++kvRow) {
        weightedSum +=
            scores[kvRow] * mergedValues[kvRow * valueWidth + outCol];
      }
      outputData[row * valueWidth + outCol] = weightedSum;
    }
  }

  return true;
#endif
}

} // namespace us4
