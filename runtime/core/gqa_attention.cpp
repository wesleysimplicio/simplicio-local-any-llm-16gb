#include "core/gqa_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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

} // namespace

bool GqaAttention(const Tensor &query, const Tensor &key, const Tensor &value,
                  const std::size_t queryHeads, const std::size_t kvHeads,
                  Tensor &output, std::string *error) {
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

  if (queryHeads == 0U || kvHeads == 0U || queryHeads < kvHeads ||
      (queryHeads % kvHeads) != 0U) {
    return WriteError(error, "invalid GQA head relationship");
  }
  if (query.Shape()[1] == 0U || (query.Shape()[1] % queryHeads) != 0U) {
    return WriteError(error,
                      "query width must divide evenly across query heads");
  }
  if (key.Shape()[0] != value.Shape()[0]) {
    return WriteError(error, "key and value token counts must match");
  }

  const std::size_t queryRows = query.Shape()[0];
  const std::size_t queryWidth = query.Shape()[1];
  const std::size_t keyRows = key.Shape()[0];
  const std::size_t keyWidth = key.Shape()[1];
  const std::size_t valueWidth = value.Shape()[1];
  const std::size_t headDim = queryWidth / queryHeads;
  if (headDim == 0U) {
    return WriteError(error, "query head dimension must be non-zero");
  }
  if (keyWidth != kvHeads * headDim) {
    return WriteError(error, "key width must equal kvHeads * headDim");
  }
  if (valueWidth != kvHeads * headDim) {
    return WriteError(error, "value width must equal kvHeads * headDim");
  }
  if (output.Shape()[0] != queryRows || output.Shape()[1] != queryWidth) {
    return WriteError(error,
                      "output shape does not match grouped attention result");
  }
  if (keyRows == 0U) {
    return WriteError(error, "attention requires at least one kv row");
  }

  const float *queryData = query.DataAsFloat32();
  const float *keyData = key.DataAsFloat32();
  const float *valueData = value.DataAsFloat32();
  float *outputData = output.MutableDataAsFloat32();
  const std::size_t queriesPerKvHead = queryHeads / kvHeads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(headDim));

  for (std::size_t queryRow = 0; queryRow < queryRows; ++queryRow) {
    for (std::size_t queryHead = 0; queryHead < queryHeads; ++queryHead) {
      const std::size_t kvHead = queryHead / queriesPerKvHead;
      const std::size_t queryBase = queryRow * queryWidth + queryHead * headDim;
      const std::size_t kvBase = kvHead * headDim;

      std::vector<float> scores(keyRows, 0.0F);
      for (std::size_t kvRow = 0; kvRow < keyRows; ++kvRow) {
        float dot = 0.0F;
        for (std::size_t col = 0; col < headDim; ++col) {
          dot += queryData[queryBase + col] *
                 keyData[kvRow * keyWidth + kvBase + col];
        }
        scores[kvRow] = dot * scale;
      }

      const float maxScore = *std::max_element(scores.begin(), scores.end());
      float denominator = 0.0F;
      for (float &score : scores) {
        score = std::exp(score - maxScore);
        denominator += score;
      }
      if (denominator <= std::numeric_limits<float>::epsilon()) {
        return WriteError(error, "attention softmax denominator is too small");
      }

      for (std::size_t col = 0; col < headDim; ++col) {
        float weightedSum = 0.0F;
        for (std::size_t kvRow = 0; kvRow < keyRows; ++kvRow) {
          weightedSum += (scores[kvRow] / denominator) *
                         valueData[kvRow * valueWidth + kvBase + col];
        }
        outputData[queryBase + col] = weightedSum;
      }
    }
  }

  return true;
}

} // namespace us4
