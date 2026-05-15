#include "neon/neon_matmul.h"

#include <algorithm>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "cpu/scalar_matmul.h"

namespace us4 {

namespace {

bool WriteError(std::string *error, const char *message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool HasReadableStorage(const Tensor &tensor) {
  switch (tensor.dtype()) {
  case DType::kFloat32:
    return tensor.DataAsFloat32() != nullptr;
  case DType::kFloat16:
  case DType::kBFloat16:
    return tensor.DataAsUInt16() != nullptr;
  case DType::kInt8:
    return tensor.Data() != nullptr;
  default:
    return false;
  }
}

bool ValidateMatrix(const Tensor &tensor, std::string *error) {
  if (tensor.dtype() != DType::kFloat32 && tensor.dtype() != DType::kFloat16 &&
      tensor.dtype() != DType::kBFloat16 && tensor.dtype() != DType::kInt8) {
    return WriteError(error, "tensor must be fp32, fp16, bf16, or int8");
  }
  if (tensor.Rank() != 2) {
    return WriteError(error, "tensor must be rank-2");
  }
  if (!tensor.IsContiguous()) {
    return WriteError(error, "tensor must be contiguous");
  }
  if (!HasReadableStorage(tensor)) {
    return WriteError(error, "tensor storage is unavailable");
  }
  return true;
}

float ReadTensorValue(const Tensor &tensor, const std::size_t index) {
  switch (tensor.dtype()) {
  case DType::kFloat32:
    return tensor.DataAsFloat32()[index];
  case DType::kFloat16:
    return DecodeFloat16(tensor.DataAsUInt16()[index]);
  case DType::kBFloat16:
    return DecodeBFloat16(tensor.DataAsUInt16()[index]);
  case DType::kInt8:
    return static_cast<float>(
        reinterpret_cast<const std::int8_t *>(tensor.Data())[index]);
  default:
    return 0.0F;
  }
}

void RunScalarLane4Matmul(const Tensor &lhs, const Tensor &rhs,
                          Tensor &output) {
  const std::size_t lhsRows = lhs.Shape()[0];
  const std::size_t lhsCols = lhs.Shape()[1];
  const std::size_t rhsCols = rhs.Shape()[1];

  float *outData = output.MutableDataAsFloat32();

  constexpr std::size_t kLaneWidth = 4U;
  for (std::size_t row = 0; row < lhsRows; ++row) {
    std::size_t col = 0;
    for (; col + kLaneWidth <= rhsCols; col += kLaneWidth) {
      float acc0 = 0.0F;
      float acc1 = 0.0F;
      float acc2 = 0.0F;
      float acc3 = 0.0F;

      for (std::size_t inner = 0; inner < lhsCols; ++inner) {
        const float lhsValue = ReadTensorValue(lhs, row * lhsCols + inner);
        const std::size_t rhsOffset = inner * rhsCols + col;
        acc0 += lhsValue * ReadTensorValue(rhs, rhsOffset + 0U);
        acc1 += lhsValue * ReadTensorValue(rhs, rhsOffset + 1U);
        acc2 += lhsValue * ReadTensorValue(rhs, rhsOffset + 2U);
        acc3 += lhsValue * ReadTensorValue(rhs, rhsOffset + 3U);
      }

      const std::size_t outOffset = row * rhsCols + col;
      outData[outOffset + 0] = acc0;
      outData[outOffset + 1] = acc1;
      outData[outOffset + 2] = acc2;
      outData[outOffset + 3] = acc3;
    }

    for (; col < rhsCols; ++col) {
      float accumulator = 0.0F;
      for (std::size_t inner = 0; inner < lhsCols; ++inner) {
        accumulator += ReadTensorValue(lhs, row * lhsCols + inner) *
                       ReadTensorValue(rhs, inner * rhsCols + col);
      }
      outData[row * rhsCols + col] = accumulator;
    }
  }
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
void RunNeonInt8Matmul(const Tensor &lhs, const Tensor &rhs, Tensor &output) {
  const std::size_t lhsRows = lhs.Shape()[0];
  const std::size_t lhsCols = lhs.Shape()[1];
  const std::size_t rhsCols = rhs.Shape()[1];

  const auto *lhsData = reinterpret_cast<const std::int8_t *>(lhs.Data());
  const auto *rhsData = reinterpret_cast<const std::int8_t *>(rhs.Data());
  float *outData = output.MutableDataAsFloat32();

  constexpr std::size_t kLaneWidth = 4U;
  for (std::size_t row = 0; row < lhsRows; ++row) {
    std::size_t col = 0;
    for (; col + kLaneWidth <= rhsCols; col += kLaneWidth) {
      int32x4_t accumulator = vdupq_n_s32(0);
      std::size_t inner = 0;

#if defined(__ARM_FEATURE_DOTPROD)
      for (; inner + kLaneWidth <= lhsCols; inner += kLaneWidth) {
        std::int8_t rhsPacked[16];
        std::int8_t lhsPacked[16];
        for (std::size_t lane = 0; lane < kLaneWidth; ++lane) {
          const std::size_t innerIndex = inner + lane;
          const std::size_t rhsOffset = innerIndex * rhsCols + col;
          rhsPacked[lane + 0U] = rhsData[rhsOffset + 0U];
          rhsPacked[lane + 4U] = rhsData[rhsOffset + 1U];
          rhsPacked[lane + 8U] = rhsData[rhsOffset + 2U];
          rhsPacked[lane + 12U] = rhsData[rhsOffset + 3U];

          const std::int8_t lhsValue = lhsData[row * lhsCols + innerIndex];
          lhsPacked[lane + 0U] = lhsValue;
          lhsPacked[lane + 4U] = lhsValue;
          lhsPacked[lane + 8U] = lhsValue;
          lhsPacked[lane + 12U] = lhsValue;
        }

        accumulator =
            vdotq_s32(accumulator, vld1q_s8(rhsPacked), vld1q_s8(lhsPacked));
      }
#endif

      for (; inner < lhsCols; ++inner) {
        const std::int32_t lhsValue =
            static_cast<std::int32_t>(lhsData[row * lhsCols + inner]);
        const std::size_t rhsOffset = inner * rhsCols + col;
        std::int8_t rhsPacked[8] = {
            rhsData[rhsOffset + 0U],
            rhsData[rhsOffset + 1U],
            rhsData[rhsOffset + 2U],
            rhsData[rhsOffset + 3U],
            0,
            0,
            0,
            0,
        };
        const int8x8_t rhsVector8 = vld1_s8(rhsPacked);
        const int16x8_t rhsWide16 = vmovl_s8(rhsVector8);
        const int32x4_t rhsWide32 = vmovl_s16(vget_low_s16(rhsWide16));
        accumulator = vmlaq_n_s32(accumulator, rhsWide32, lhsValue);
      }

      vst1q_f32(outData + (row * rhsCols + col), vcvtq_f32_s32(accumulator));
    }

    for (; col < rhsCols; ++col) {
      std::int32_t accumulator = 0;
      for (std::size_t inner = 0; inner < lhsCols; ++inner) {
        accumulator +=
            static_cast<std::int32_t>(lhsData[row * lhsCols + inner]) *
            static_cast<std::int32_t>(rhsData[inner * rhsCols + col]);
      }
      outData[row * rhsCols + col] = static_cast<float>(accumulator);
    }
  }
}

void RunNeonLane4Matmul(const Tensor &lhs, const Tensor &rhs, Tensor &output) {
  if (lhs.dtype() == DType::kInt8 && rhs.dtype() == DType::kInt8) {
    RunNeonInt8Matmul(lhs, rhs, output);
    return;
  }

  if (lhs.dtype() != DType::kFloat32 || rhs.dtype() != DType::kFloat32) {
    RunScalarLane4Matmul(lhs, rhs, output);
    return;
  }

  const std::size_t lhsRows = lhs.Shape()[0];
  const std::size_t lhsCols = lhs.Shape()[1];
  const std::size_t rhsCols = rhs.Shape()[1];

  const float *lhsData = lhs.DataAsFloat32();
  const float *rhsData = rhs.DataAsFloat32();
  float *outData = output.MutableDataAsFloat32();

  constexpr std::size_t kLaneWidth = 4U;
  for (std::size_t row = 0; row < lhsRows; ++row) {
    std::size_t col = 0;
    for (; col + kLaneWidth <= rhsCols; col += kLaneWidth) {
      float32x4_t accumulator = vdupq_n_f32(0.0F);

      for (std::size_t inner = 0; inner < lhsCols; ++inner) {
        const float lhsValue = lhsData[row * lhsCols + inner];
        const float32x4_t rhsVector =
            vld1q_f32(rhsData + (inner * rhsCols + col));
        accumulator = vaddq_f32(accumulator, vmulq_n_f32(rhsVector, lhsValue));
      }

      vst1q_f32(outData + (row * rhsCols + col), accumulator);
    }

    for (; col < rhsCols; ++col) {
      float accumulator = 0.0F;
      for (std::size_t inner = 0; inner < lhsCols; ++inner) {
        accumulator +=
            lhsData[row * lhsCols + inner] * rhsData[inner * rhsCols + col];
      }
      outData[row * rhsCols + col] = accumulator;
    }
  }
}
#endif

} // namespace

bool NeonMatmul(const Tensor &lhs, const Tensor &rhs, Tensor &output,
                std::string *error) {
  if (!ValidateMatrix(lhs, error) || !ValidateMatrix(rhs, error)) {
    return false;
  }

  if (output.dtype() != DType::kFloat32 || output.Rank() != 2 ||
      output.MutableDataAsFloat32() == nullptr || !output.IsContiguous()) {
    return WriteError(error, "output must be a writable fp32 rank-2 tensor");
  }

  const std::size_t lhsRows = lhs.Shape()[0];
  const std::size_t lhsCols = lhs.Shape()[1];
  const std::size_t rhsRows = rhs.Shape()[0];
  const std::size_t rhsCols = rhs.Shape()[1];

  if (lhsCols != rhsRows) {
    return WriteError(error, "lhs columns must match rhs rows");
  }
  if (output.Shape()[0] != lhsRows || output.Shape()[1] != rhsCols) {
    return WriteError(error, "output shape does not match matmul result");
  }

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  RunNeonLane4Matmul(lhs, rhs, output);
#else
  RunScalarLane4Matmul(lhs, rhs, output);
#endif

  return true;
}

} // namespace us4
