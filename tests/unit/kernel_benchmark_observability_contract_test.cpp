#include <gtest/gtest.h>

#include "benchmarks/kernel_benchmark_observability.h"

TEST(KernelBenchmarkObservabilityContractTest,
     NeonMatmulObservationShowsFallbackWhenProfileStaysScalarBridge) {
  us4::NeonMatmulProfile profile;
  profile.flavor = us4::NeonKernelFlavor::kScalarBridge;

  const us4::benchmarks::KernelObservation observation =
      us4::benchmarks::ObserveKernelMatmul(us4::BackendType::kNeon, profile);

  EXPECT_EQ(observation.requestedBackend, "neon");
  EXPECT_EQ(observation.observedBackend, "scalar");
  EXPECT_EQ(observation.backendReason, "requested-backend-unavailable");
  EXPECT_EQ(observation.kernelFlavor, "scalar-bridge");
}

TEST(KernelBenchmarkObservabilityContractTest,
     NeonAttentionObservationShowsRealNeonExecutionWhenPlanned) {
  us4::NeonAttentionProfile profile;
  profile.flavor = us4::NeonKernelFlavor::kFp32Lane4;
  profile.vectorLanes = 4U;

  const us4::benchmarks::KernelObservation observation =
      us4::benchmarks::ObserveKernelAttention(us4::BackendType::kNeon, profile);

  EXPECT_EQ(observation.requestedBackend, "neon");
  EXPECT_EQ(observation.observedBackend, "neon");
  EXPECT_EQ(observation.backendReason, "requested");
  EXPECT_EQ(observation.kernelFlavor, "fp32-lane4");
}

TEST(KernelBenchmarkObservabilityContractTest,
     OperationCountersAndThroughputStayDeterministic) {
  EXPECT_DOUBLE_EQ(us4::benchmarks::MatmulOperations(128U, 256U, 128U),
                   8388608.0);
  EXPECT_DOUBLE_EQ(us4::benchmarks::AttentionOperations(1024U, 64U, 64U),
                   262144.0);
  EXPECT_DOUBLE_EQ(us4::benchmarks::ThroughputGops(1'000'000'000.0, 1000.0),
                   1.0);
}
