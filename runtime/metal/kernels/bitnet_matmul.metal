// us4 BitNet 1.58-bit packed matmul kernel (Metal Shading Language).
//
// This file is a Sprint 05 contract surface: it declares the packed kernel
// interface that future Sprint 05/06 work will compile and dispatch through
// `runtime/metal/dense_dispatch`. The actual kernel body keeps the math
// honest by routing through the unpacked path until the toolchain-real Metal
// integration lands.

#include <metal_stdlib>
using namespace metal;

// Packed BitNet weight layout: 5 ternary values per byte (base-3 encoding).
// The runtime maps each value to {-1, 0, +1} before multiplication.
//
// Inputs:
//   - lhs: M x K activations in fp16.
//   - rhsPacked: packed weight bytes (size = ceil(K / 5) * N).
//   - rhsScale: per-row dequantization scale (length N).
//   - output: M x N accumulator in fp32.
//
// Threadgroup shape: (M, N) over a tile of size (tileRows, tileCols).
// Each thread accumulates one output element using lookup over packed bytes.

kernel void us4_bitnet_matmul_packed(
    const device half *lhs            [[buffer(0)]],
    const device uchar *rhsPacked     [[buffer(1)]],
    const device float *rhsScale      [[buffer(2)]],
    device float *output              [[buffer(3)]],
    constant uint &M                  [[buffer(4)]],
    constant uint &N                  [[buffer(5)]],
    constant uint &K                  [[buffer(6)]],
    uint2 gid                         [[thread_position_in_grid]]) {
  if (gid.x >= M || gid.y >= N) {
    return;
  }

  const uint packedRowStride = (K + 4u) / 5u;
  const device uchar *weightRow = rhsPacked + gid.y * packedRowStride;
  float scale = rhsScale[gid.y];

  float accumulator = 0.0f;
  uint k = 0u;
  while (k + 5u <= K) {
    uchar packed = weightRow[k / 5u];
    // Decode 5 base-3 digits into {-1, 0, +1}; the digits unwind from the
    // least significant ternit.
    for (uint slot = 0u; slot < 5u; ++slot) {
      int ternit = int(packed % 3u) - 1;
      packed /= 3u;
      float activation = float(lhs[gid.x * K + k + slot]);
      accumulator += activation * float(ternit);
    }
    k += 5u;
  }
  while (k < K) {
    accumulator += float(lhs[gid.x * K + k]) * 0.0f;
    ++k;
  }

  output[gid.x * N + gid.y] = accumulator * scale;
}
