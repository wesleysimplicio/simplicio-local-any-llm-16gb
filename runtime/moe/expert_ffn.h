#pragma once

#include <vector>

#include "core/model_asset.h"

namespace us4 {

// Real per-expert MLP forward (issue #81.7c): the SwiGLU feed-forward
// layer used by Mixtral/DeepSeek-style MoE experts --
// down_proj(silu(gate_proj(x)) * up_proj(x)) -- applied to the attention
// context `x` (size `hiddenSize`) using a specific expert's REAL weights,
// instead of only swapping the shared output projection (#81.7/#81.7b).
// Returns a vector of size `hiddenSize` (the same shape `x` came in as).
// Callers are responsible for verifying `weights`' tensor shapes already
// match `hiddenSize`/the intended intermediate size (see
// TryLoadExpertShardFfn), since this function itself does no bounds
// checking -- it's meant to run over trusted, already-validated tensors.
std::vector<float> ApplyExpertFfnSwiglu(const std::vector<float> &x,
                                        const ExpertFfnWeights &weights);

} // namespace us4
