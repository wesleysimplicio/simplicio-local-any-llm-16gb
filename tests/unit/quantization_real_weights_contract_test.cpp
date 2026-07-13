#include <cmath>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "core/safetensors_reader.h"
#include "core/tensor.h"
#include "cpu/quantize_projection.h"
#include "neon/dequant_int4.h"
#include "neon/dequant_int8.h"

namespace us4 {
namespace {

constexpr std::size_t kGroupSize = 8U;

std::filesystem::path FixtureDir() {
  return std::filesystem::path(US4_SOURCE_DIR) / "tests" / "fixtures" /
         "safetensors";
}

std::vector<float> LoadRealTensor(const std::string &name) {
  std::string error;
  const auto reader =
      SafetensorsReader::Open(FixtureDir() / "toy_real.safetensors", &error);
  if (!reader.has_value()) {
    ADD_FAILURE() << "failed to open fixture: " << error;
    return {};
  }
  std::vector<float> values = reader->ReadFloat32(name, &error);
  if (values.empty()) {
    ADD_FAILURE() << "failed to read tensor " << name << ": " << error;
  }
  return values;
}

} // namespace

// Issue #81.8: quantization must be validated over REAL weights loaded from
// a genuine safetensors file (issue #81.2), not over the synthetic
// projection the dense forward builds when no real weights are present.
TEST(QuantizationRealWeightsContractTest,
     Int8RoundTripStaysWithinDocumentedToleranceOnRealWeights) {
  const std::vector<float> realWeights = LoadRealTensor("embedding.weight");
  ASSERT_FALSE(realWeights.empty());

  const GroupwiseQuantizedProjection quantized =
      QuantizeProjectionInt8(realWeights, {realWeights.size()}, kGroupSize);
  ASSERT_EQ(quantized.tensor.dtype(), DType::kInt8);

  Tensor dequantized({realWeights.size()}, DType::kFloat32);
  std::string error;
  ASSERT_TRUE(DequantizeInt8Groups(quantized.tensor, kGroupSize,
                                   quantized.scales, dequantized, &error))
      << error;

  const float *recovered = dequantized.DataAsFloat32();
  for (std::size_t index = 0; index < realWeights.size(); ++index) {
    const float scale = quantized.scales[index / kGroupSize];
    // Documented tolerance: int8 rounding to the nearest quantization step
    // can be off by at most half a step; one full `scale` gives headroom
    // for accumulated floating point error while still catching any real
    // regression in the quantizer/dequantizer.
    EXPECT_NEAR(recovered[index], realWeights[index], scale)
        << "element " << index << " (scale=" << scale << ")";
  }
}

TEST(QuantizationRealWeightsContractTest,
     Int4RoundTripStaysWithinDocumentedToleranceOnRealWeights) {
  const std::vector<float> realWeights = LoadRealTensor("lm_head.weight");
  ASSERT_FALSE(realWeights.empty());

  const GroupwiseQuantizedProjection quantized =
      QuantizeProjectionInt4(realWeights, {realWeights.size()}, kGroupSize);
  ASSERT_EQ(quantized.tensor.dtype(), DType::kInt4);

  Tensor dequantized({realWeights.size()}, DType::kFloat32);
  std::string error;
  ASSERT_TRUE(DequantizeInt4Groups(quantized.tensor, realWeights.size(),
                                   kGroupSize, quantized.scales, dequantized,
                                   &error))
      << error;

  const float *recovered = dequantized.DataAsFloat32();
  for (std::size_t index = 0; index < realWeights.size(); ++index) {
    const float scale = quantized.scales[index / kGroupSize];
    // int4 has a much coarser step (16 levels vs 254 for int8), so the
    // documented tolerance is wider: up to one full step either side.
    EXPECT_NEAR(recovered[index], realWeights[index], scale)
        << "element " << index << " (scale=" << scale << ")";
  }
}

} // namespace us4
