#pragma once

#include <cstddef>

#include "core/model_asset.h"
#include "core/rope.h"

namespace us4 {

struct LlamaConfig {
  std::size_t hiddenSize = 8U;
  std::size_t queryHeads = 2U;
  std::size_t kvHeads = 1U;
  std::size_t headDim = 4U;
  float ropeTheta = 10000.0F;
  RopeScalingType ropeScaling = RopeScalingType::kDynamic;
  float ropeScale = 1.0F;
};

LlamaConfig ResolveLlamaConfig(const ModelAsset *asset);

} // namespace us4
