#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "benchmarks/kernel_benchmark_observability.h"
#include "core/hardware_probe.h"
#include "core/tensor.h"
#include "cpu/scalar_attention.h"
#include "cpu/scalar_matmul.h"
#include "neon/kernel_profile.h"
#include "neon/neon_attention.h"
#include "neon/neon_matmul.h"

namespace {

using Clock = std::chrono::steady_clock;

template <typename Fn>
std::vector<double> MeasureRepeated(const std::size_t repeats, Fn &&fn) {
  std::vector<double> samples;
  samples.reserve(repeats);
  for (std::size_t iteration = 0; iteration < repeats; ++iteration) {
    const auto start = Clock::now();
    fn();
    const auto end = Clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start)
                          .count());
  }
  return samples;
}

double MeanMs(const std::vector<double> &samples) {
  if (samples.empty()) {
    return 0.0;
  }
  return std::accumulate(samples.begin(), samples.end(), 0.0) /
         static_cast<double>(samples.size());
}

double MinMs(const std::vector<double> &samples) {
  return samples.empty() ? 0.0
                         : *std::min_element(samples.begin(), samples.end());
}

double MaxMs(const std::vector<double> &samples) {
  return samples.empty() ? 0.0
                         : *std::max_element(samples.begin(), samples.end());
}

void FillFloatTensor(us4::Tensor &tensor, const float scale, const float bias) {
  float *data = tensor.MutableDataAsFloat32();
  for (std::size_t index = 0; index < tensor.ElementCount(); ++index) {
    const int centered = static_cast<int>(index % 17U) - 8;
    data[index] = static_cast<float>(centered) * scale + bias;
  }
}

void FillHalfTensor(us4::Tensor &tensor, const float scale, const float bias,
                    const bool bfloat16) {
  std::uint16_t *data = tensor.MutableDataAsUInt16();
  for (std::size_t index = 0; index < tensor.ElementCount(); ++index) {
    const int centered = static_cast<int>(index % 19U) - 9;
    const float value = static_cast<float>(centered) * scale + bias;
    data[index] = bfloat16 ? us4::EncodeBFloat16(value)
                           : us4::EncodeFloat16(value);
  }
}

void FillInt8Tensor(us4::Tensor &tensor) {
  auto *data = reinterpret_cast<std::int8_t *>(tensor.MutableData());
  for (std::size_t index = 0; index < tensor.ElementCount(); ++index) {
    const int centered = static_cast<int>(index % 15U) - 7;
    data[index] = static_cast<std::int8_t>(centered);
  }
}

double SumTensor(const us4::Tensor &tensor) {
  const float *data = tensor.DataAsFloat32();
  return std::accumulate(data, data + tensor.ElementCount(), 0.0);
}

struct MatmulCase {
  const char *label;
  us4::DType dtype;
  std::optional<us4::BackendType> requestedBackend;
  std::size_t lhsRows;
  std::size_t lhsCols;
  std::size_t rhsCols;
};

struct AttentionCase {
  const char *label;
  std::optional<us4::BackendType> requestedBackend;
  std::size_t queryRows;
  std::size_t kvRows;
  std::size_t hidden;
  std::size_t valueWidth;
  bool causalMask;
};

void PrintHeader(const us4::HardwareProbeResult &probe) {
  std::cout << "benchmark=kernel_cpu_benchmarks\n";
  std::cout << "platform=" << probe.platform << "\n";
  std::cout << "architecture=" << probe.architecture << "\n";
  std::cout << "chip=" << probe.chip << "\n";
  std::cout << "has_neon=" << (probe.hasNeon ? "true" : "false") << "\n";
  std::cout << "neon_vector_bits=" << probe.neonVectorBits << "\n";
  std::cout << "note=metal-mlx-benchmarks-blocked-without-apple-silicon\n";
  std::cout << "--\n";
}

const char *DTypeName(const us4::DType dtype) {
  switch (dtype) {
  case us4::DType::kFloat32:
    return "fp32";
  case us4::DType::kFloat16:
    return "fp16";
  case us4::DType::kBFloat16:
    return "bf16";
  case us4::DType::kInt8:
    return "int8";
  case us4::DType::kInt4:
    return "int4";
  }
  return "unknown";
}

void PrintMatmulRow(const us4::HardwareProbeResult &probe,
                    const MatmulCase &benchmarkCase) {
  us4::Tensor lhs({benchmarkCase.lhsRows, benchmarkCase.lhsCols},
                  benchmarkCase.dtype, us4::DeviceType::kCpu);
  us4::Tensor rhs({benchmarkCase.lhsCols, benchmarkCase.rhsCols},
                  benchmarkCase.dtype, us4::DeviceType::kCpu);
  us4::Tensor output({benchmarkCase.lhsRows, benchmarkCase.rhsCols},
                     us4::DType::kFloat32, us4::DeviceType::kCpu);

  switch (benchmarkCase.dtype) {
  case us4::DType::kFloat32:
    FillFloatTensor(lhs, 0.0625F, 0.25F);
    FillFloatTensor(rhs, 0.03125F, -0.125F);
    break;
  case us4::DType::kFloat16:
    FillHalfTensor(lhs, 0.0625F, 0.25F, false);
    FillHalfTensor(rhs, 0.03125F, -0.125F, false);
    break;
  case us4::DType::kBFloat16:
    FillHalfTensor(lhs, 0.0625F, 0.25F, true);
    FillHalfTensor(rhs, 0.03125F, -0.125F, true);
    break;
  case us4::DType::kInt8:
    FillInt8Tensor(lhs);
    FillInt8Tensor(rhs);
    break;
  case us4::DType::kInt4:
    return;
  }

  const std::size_t repeats = 10U;
  const us4::NeonMatmulProfile profile = us4::PlanNeonMatmul(probe, lhs, rhs);
  const us4::benchmarks::KernelObservation observation =
      us4::benchmarks::ObserveKernelMatmul(benchmarkCase.requestedBackend,
                                           profile);

  std::string error;
  const auto runner = [&]() {
    if (benchmarkCase.requestedBackend == us4::BackendType::kScalarCpu) {
      return us4::ScalarMatmul(lhs, rhs, output, &error);
    }
    return us4::NeonMatmul(lhs, rhs, output, &error);
  };

  if (!runner()) {
    std::cerr << "matmul benchmark failed for " << benchmarkCase.label << ": "
              << error << "\n";
    std::exit(1);
  }

  const std::vector<double> samples =
      MeasureRepeated(repeats, [&]() { (void)runner(); });
  const double meanMs = MeanMs(samples);
  const double operations = us4::benchmarks::MatmulOperations(
      benchmarkCase.lhsRows, benchmarkCase.lhsCols, benchmarkCase.rhsCols);

  std::cout << "case=" << benchmarkCase.label << "\n";
  std::cout << "kernel=matmul\n";
  std::cout << "dtype=" << DTypeName(benchmarkCase.dtype) << "\n";
  std::cout << "requested_backend=" << observation.requestedBackend << "\n";
  std::cout << "observed_backend=" << observation.observedBackend << "\n";
  std::cout << "backend_reason=" << observation.backendReason << "\n";
  std::cout << "kernel_flavor=" << observation.kernelFlavor << "\n";
  std::cout << "tile_rows=" << profile.tileRows << "\n";
  std::cout << "tile_cols=" << profile.tileCols << "\n";
  std::cout << "vector_lanes=" << profile.vectorLanes << "\n";
  std::cout << "uses_dot_product="
            << (profile.usesDotProduct ? "true" : "false") << "\n";
  std::cout << "uses_accelerate_fallback="
            << (profile.usesAccelerateFallback ? "true" : "false") << "\n";
  std::cout << "lhs_rows=" << benchmarkCase.lhsRows << "\n";
  std::cout << "lhs_cols=" << benchmarkCase.lhsCols << "\n";
  std::cout << "rhs_cols=" << benchmarkCase.rhsCols << "\n";
  std::cout << "repeats=" << repeats << "\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "elapsed_ms_mean=" << meanMs << "\n";
  std::cout << "elapsed_ms_min=" << MinMs(samples) << "\n";
  std::cout << "elapsed_ms_max=" << MaxMs(samples) << "\n";
  std::cout << "throughput_gops="
            << us4::benchmarks::ThroughputGops(operations, meanMs) << "\n";
  std::cout << "output_checksum=" << SumTensor(output) << "\n";
  std::cout << "--\n";
}

void PrintAttentionRow(const us4::HardwareProbeResult &probe,
                       const AttentionCase &benchmarkCase) {
  us4::Tensor query({benchmarkCase.queryRows, benchmarkCase.hidden},
                    us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor key({benchmarkCase.kvRows, benchmarkCase.hidden},
                  us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor value({benchmarkCase.kvRows, benchmarkCase.valueWidth},
                    us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor output({benchmarkCase.queryRows, benchmarkCase.valueWidth},
                     us4::DType::kFloat32, us4::DeviceType::kCpu);

  FillFloatTensor(query, 0.050F, -0.125F);
  FillFloatTensor(key, 0.040F, 0.0625F);
  FillFloatTensor(value, 0.035F, -0.03125F);

  const std::size_t repeats = 10U;
  const us4::NeonAttentionProfile profile = us4::PlanNeonAttention(
      probe, query, key, value, benchmarkCase.causalMask);
  const us4::benchmarks::KernelObservation observation =
      us4::benchmarks::ObserveKernelAttention(
          benchmarkCase.requestedBackend, profile);

  std::string error;
  const auto runner = [&]() {
    if (benchmarkCase.requestedBackend == us4::BackendType::kScalarCpu) {
      return us4::ScalarAttention(query, key, value, output,
                                  benchmarkCase.causalMask, {}, &error);
    }
    return us4::NeonAttention(query, key, value, output,
                              benchmarkCase.causalMask, {}, &error);
  };

  if (!runner()) {
    std::cerr << "attention benchmark failed for " << benchmarkCase.label
              << ": " << error << "\n";
    std::exit(1);
  }

  const std::vector<double> samples =
      MeasureRepeated(repeats, [&]() { (void)runner(); });
  const double meanMs = MeanMs(samples);
  const std::size_t totalVisibleRows = benchmarkCase.causalMask
                                           ? ((benchmarkCase.queryRows *
                                               (benchmarkCase.queryRows + 1U)) /
                                              2U)
                                           : (benchmarkCase.queryRows *
                                              benchmarkCase.kvRows);
  const double operations = us4::benchmarks::AttentionOperations(
      totalVisibleRows, benchmarkCase.hidden, benchmarkCase.valueWidth);

  std::cout << "case=" << benchmarkCase.label << "\n";
  std::cout << "kernel=attention\n";
  std::cout << "dtype=fp32\n";
  std::cout << "requested_backend=" << observation.requestedBackend << "\n";
  std::cout << "observed_backend=" << observation.observedBackend << "\n";
  std::cout << "backend_reason=" << observation.backendReason << "\n";
  std::cout << "kernel_flavor=" << observation.kernelFlavor << "\n";
  std::cout << "vector_lanes=" << profile.vectorLanes << "\n";
  std::cout << "head_dim_block=" << profile.headDimBlock << "\n";
  std::cout << "fuses_softmax_rescale="
            << (profile.fusesSoftmaxRescale ? "true" : "false") << "\n";
  std::cout << "supports_causal_mask="
            << (profile.supportsCausalMask ? "true" : "false") << "\n";
  std::cout << "query_rows=" << benchmarkCase.queryRows << "\n";
  std::cout << "kv_rows=" << benchmarkCase.kvRows << "\n";
  std::cout << "hidden=" << benchmarkCase.hidden << "\n";
  std::cout << "value_width=" << benchmarkCase.valueWidth << "\n";
  std::cout << "causal_mask="
            << (benchmarkCase.causalMask ? "true" : "false") << "\n";
  std::cout << "repeats=" << repeats << "\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "elapsed_ms_mean=" << meanMs << "\n";
  std::cout << "elapsed_ms_min=" << MinMs(samples) << "\n";
  std::cout << "elapsed_ms_max=" << MaxMs(samples) << "\n";
  std::cout << "throughput_gops="
            << us4::benchmarks::ThroughputGops(operations, meanMs) << "\n";
  std::cout << "output_checksum=" << SumTensor(output) << "\n";
  std::cout << "--\n";
}

} // namespace

int main() {
  const us4::HardwareProbeResult probe = us4::HardwareProbe::Detect();
  PrintHeader(probe);

  const std::array<MatmulCase, 5> matmulCases = {{
      {"matmul-fp32/scalar", us4::DType::kFloat32,
       us4::BackendType::kScalarCpu, 128U, 256U, 128U},
      {"matmul-fp32/neon", us4::DType::kFloat32, us4::BackendType::kNeon, 128U,
       256U, 128U},
      {"matmul-fp16/neon", us4::DType::kFloat16, us4::BackendType::kNeon, 128U,
       256U, 128U},
      {"matmul-bf16/neon", us4::DType::kBFloat16, us4::BackendType::kNeon,
       128U, 256U, 128U},
      {"matmul-int8/neon", us4::DType::kInt8, us4::BackendType::kNeon, 128U,
       256U, 128U},
  }};

  for (const MatmulCase &benchmarkCase : matmulCases) {
    PrintMatmulRow(probe, benchmarkCase);
  }

  const std::array<AttentionCase, 3> attentionCases = {{
      {"attention-fp32/scalar", us4::BackendType::kScalarCpu, 8U, 128U, 64U,
       64U, false},
      {"attention-fp32/neon", us4::BackendType::kNeon, 8U, 128U, 64U, 64U,
       false},
      {"attention-fp32/neon-causal", us4::BackendType::kNeon, 8U, 128U, 64U,
       64U, true},
  }};

  for (const AttentionCase &benchmarkCase : attentionCases) {
    PrintAttentionRow(probe, benchmarkCase);
  }

  return 0;
}
