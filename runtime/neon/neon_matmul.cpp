#include "neon/neon_matmul.h"

#include "cpu/scalar_matmul.h"

namespace us4 {

bool NeonMatmul(const Tensor& lhs, const Tensor& rhs, Tensor& output, std::string* error) {
  // Sprint 04 will replace this reference shim with real NEON kernels.
  return ScalarMatmul(lhs, rhs, output, error);
}

}  // namespace us4
