#include "benchmarks/neon_lowbit_observability.h"

#include <gtest/gtest.h>

namespace {

us4::GenerationResult MakeResult(const std::string_view backend,
                                 const std::string_view backendReason,
                                 const std::string_view weightDType,
                                 const std::string_view neonKernelFlavor,
                                 const std::string_view dequantPath,
                                 const std::string_view text,
                                 const bool fellBack) {
  us4::GenerationResult result;
  result.backend = std::string(backend);
  result.backendReason = std::string(backendReason);
  result.weightDType = std::string(weightDType);
  result.neonKernelFlavor = std::string(neonKernelFlavor);
  result.dequantPath = std::string(dequantPath);
  result.generatedTokens = {"hello", "world"};
  result.text = std::string(text);
  result.fellBack = fellBack;
  return result;
}

} // namespace

TEST(BenchmarkObservabilityContractTest,
     ObserveCaseCapturesRequestedBackendAndFingerprint) {
  const us4::GenerationResult result =
      MakeResult("neon", "requested", "int8", "int8-dot", "groupwise-int8",
                 "hello world", false);

  const us4::benchmarks::CaseObservation observation =
      us4::benchmarks::ObserveCase("lowbit-int8/neon", result, 12,
                                   us4::BackendType::kNeon);

  EXPECT_EQ(observation.label, "lowbit-int8/neon");
  EXPECT_EQ(observation.requestedBackend, "neon");
  EXPECT_EQ(observation.observedBackend, "neon");
  EXPECT_EQ(observation.textFingerprint,
            us4::benchmarks::FingerprintText("hello world"));
  EXPECT_EQ(observation.elapsedMs, 12);
  EXPECT_FALSE(observation.fellBack);
}

TEST(BenchmarkObservabilityContractTest,
     CompareLowBitObservationsPassesOnMatchingNeonExecution) {
  const us4::benchmarks::CaseObservation scalarObservation =
      us4::benchmarks::ObserveCase(
          "lowbit-int8/scalar",
          MakeResult("scalar", "requested", "int8", "none", "groupwise-int8",
                     "same output", false),
          20, us4::BackendType::kScalarCpu);
  const us4::benchmarks::CaseObservation neonObservation =
      us4::benchmarks::ObserveCase(
          "lowbit-int8/neon",
          MakeResult("neon", "requested", "int8", "int8-dot", "groupwise-int8",
                     "same output", false),
          10, us4::BackendType::kNeon);

  const us4::benchmarks::LowBitRegression regression =
      us4::benchmarks::CompareLowBitObservations(scalarObservation,
                                                 neonObservation);

  EXPECT_EQ(regression.status, "pass");
  EXPECT_EQ(regression.reason, "outputs-match-and-neon-path-visible");
  EXPECT_TRUE(regression.textMatch);
  EXPECT_TRUE(regression.dequantPathMatch);
  EXPECT_TRUE(regression.neonExecuted);
  EXPECT_DOUBLE_EQ(regression.speedupVsScalar, 2.0);
}

TEST(BenchmarkObservabilityContractTest,
     CompareLowBitObservationsWarnsOnMatchingFallback) {
  const us4::benchmarks::CaseObservation scalarObservation =
      us4::benchmarks::ObserveCase(
          "lowbit-int4/scalar",
          MakeResult("scalar", "requested", "int4", "none", "groupwise-int4",
                     "same output", false),
          15, us4::BackendType::kScalarCpu);
  const us4::benchmarks::CaseObservation neonObservation =
      us4::benchmarks::ObserveCase(
          "lowbit-int4/neon",
          MakeResult("scalar", "requested-backend-unavailable", "int4",
                     "none", "groupwise-int4", "same output", true),
          15, us4::BackendType::kNeon);

  const us4::benchmarks::LowBitRegression regression =
      us4::benchmarks::CompareLowBitObservations(scalarObservation,
                                                 neonObservation);

  EXPECT_EQ(regression.status, "warn");
  EXPECT_EQ(regression.reason,
            "requested-neon-fell-back-with-matching-output");
  EXPECT_TRUE(regression.fallbackObserved);
  EXPECT_FALSE(regression.neonExecuted);
}

TEST(BenchmarkObservabilityContractTest,
     CompareLowBitObservationsFailsOnTextDrift) {
  const us4::benchmarks::CaseObservation scalarObservation =
      us4::benchmarks::ObserveCase(
          "lowbit-int8/scalar",
          MakeResult("scalar", "requested", "int8", "none", "groupwise-int8",
                     "alpha beta", false),
          18, us4::BackendType::kScalarCpu);
  const us4::benchmarks::CaseObservation neonObservation =
      us4::benchmarks::ObserveCase(
          "lowbit-int8/neon",
          MakeResult("neon", "requested", "int8", "int8-dot", "groupwise-int8",
                     "alpha gamma", false),
          9, us4::BackendType::kNeon);

  const us4::benchmarks::LowBitRegression regression =
      us4::benchmarks::CompareLowBitObservations(scalarObservation,
                                                 neonObservation);

  EXPECT_EQ(regression.status, "fail");
  EXPECT_EQ(regression.reason, "output-or-metadata-drift");
  EXPECT_FALSE(regression.textMatch);
}
