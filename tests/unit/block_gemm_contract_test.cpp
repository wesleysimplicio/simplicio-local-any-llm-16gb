#include <gtest/gtest.h>

#include "core/hardware_probe.h"
#include "core/tensor.h"
#include "neon/block_gemm.h"

namespace {

us4::HardwareProbeResult MakeNeonHardware() {
  us4::HardwareProbeResult hardware;
  hardware.hasNeon = true;
  hardware.architecture = "arm64";
  hardware.neonVectorBits = 128;
  return hardware;
}

us4::HardwareProbeResult MakeNonArmHardware() {
  us4::HardwareProbeResult hardware;
  hardware.hasNeon = false;
  hardware.architecture = "x86_64";
  hardware.neonVectorBits = 0;
  return hardware;
}

us4::Tensor MakeMatrix(const std::vector<std::size_t> &shape) {
  return us4::Tensor(shape, us4::DType::kFloat32);
}

} // namespace

TEST(BlockGemmContractTest, TileShapeIsCacheAwareOnArmHosts) {
  const auto hardware = MakeNeonHardware();
  const auto lhs = MakeMatrix({16, 16});
  const auto rhs = MakeMatrix({16, 16});

  const auto shape = us4::SelectBlockGemmTileShape(hardware, lhs, rhs);

  EXPECT_TRUE(shape.cacheAware);
  EXPECT_GE(shape.rows, 1U);
  EXPECT_GE(shape.cols, 1U);
  EXPECT_GE(shape.inner, 1U);
  EXPECT_GE(shape.prefetchDistance, 16U);
}

TEST(BlockGemmContractTest, TileShapeFallsBackOnNonArmHosts) {
  const auto hardware = MakeNonArmHardware();
  const auto lhs = MakeMatrix({16, 16});
  const auto rhs = MakeMatrix({16, 16});

  const auto shape = us4::SelectBlockGemmTileShape(hardware, lhs, rhs);

  EXPECT_FALSE(shape.cacheAware);
  EXPECT_EQ(shape.rows, 0U);
  EXPECT_EQ(shape.cols, 0U);
}

TEST(BlockGemmContractTest, NarrowMatrixPrefersWideTile) {
  const auto hardware = MakeNeonHardware();
  const auto lhs = MakeMatrix({2, 16});
  const auto rhs = MakeMatrix({16, 32});

  const auto shape = us4::SelectBlockGemmTileShape(hardware, lhs, rhs);

  EXPECT_GE(shape.cols, shape.rows);
}

TEST(BlockGemmContractTest, FormatProducesStableDescriptor) {
  const auto hardware = MakeNeonHardware();
  const auto lhs = MakeMatrix({8, 8});
  const auto rhs = MakeMatrix({8, 16});
  const auto shape = us4::SelectBlockGemmTileShape(hardware, lhs, rhs);

  char scratch[64] = {};
  const auto formatted =
      us4::FormatBlockGemmTileShape(shape, scratch, sizeof(scratch));

  EXPECT_FALSE(formatted.empty());
  EXPECT_NE(formatted.find('x'), std::string_view::npos);
  EXPECT_NE(formatted.find("prefetch="), std::string_view::npos);
}
