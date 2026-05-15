#include "neon/neon_attention.h"

#include "cpu/scalar_attention.h"

namespace us4 {

bool NeonAttention(const Tensor& query,
                   const Tensor& key,
                   const Tensor& value,
                   Tensor& output,
                   const bool causalMask,
                   const AttentionCacheView cache,
                   std::string* error) {
  // Sprint 04 will replace this reference shim with real NEON kernels.
  return ScalarAttention(query, key, value, output, causalMask, cache, error);
}

}  // namespace us4
