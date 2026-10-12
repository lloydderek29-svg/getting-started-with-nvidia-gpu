#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

namespace ch04 {

constexpr int kBins = 256;

inline int sm_count() {
  static const int sms = [] {
    int dev = 0, v = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &v, cudaDevAttrMultiProcessorCount, dev));
    return v;
  }();
  return sms;
}

// Version 0: every thread updates global memory.
__global__ void hist_global(const uint8_t* __restrict__ in,
                            unsigned* __restrict__ bins,
                            size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    atomicAdd(&bins[in[i]], 1u);
  }
}

inline void launch_hist_global(const uint8_t* d_in,
                               unsigned* d_bins, size_t n,
                               cudaStream_t s = 0) {
  hist_global<<<sm_count() * 8, 256, 0, s>>>(d_in, d_bins, n);
}

}  // namespace ch04

// Version 1: per-block histogram in shared memory.
__global__ void hist_shared(const uint8_t* __restrict__ in,
                            unsigned* __restrict__ bins,
                            size_t n) {
  __shared__ unsigned local[kBins];
  for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
    local[b] = 0;
  }
  __syncthreads();

  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    atomicAdd(&local[in[i]], 1u);
  }
  __syncthreads();

  for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
    if (local[b]) atomicAdd(&bins[b], local[b]);
  }
}

inline void launch_hist_shared(const uint8_t* d_in,
                               unsigned* d_bins, size_t n,
                               cudaStream_t s = 0) {
  hist_shared<<<sm_count() * 8, 256, 0, s>>>(d_in, d_bins, n);
}

__device__ __forceinline__ void count_word(unsigned* local,
                                           unsigned w) {
  atomicAdd(&local[w & 0xffu], 1u);
  atomicAdd(&local[(w >> 8) & 0xffu], 1u);
  atomicAdd(&local[(w >> 16) & 0xffu], 1u);
  atomicAdd(&local[w >> 24], 1u);
}

// Version 2: 16-byte loads. The input must be 16-byte aligned,
// which cudaMalloc guarantees.
__global__ void hist_shared_vec(
    const uint8_t* __restrict__ in,
    unsigned* __restrict__ bins, size_t n) {
  __shared__ unsigned local[kBins];
  for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
    local[b] = 0;
  }
  __syncthreads();

  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t nvec = n / 16;
  const uint4* in4 = reinterpret_cast<const uint4*>(in);

  for (size_t i = tid; i < nvec; i += stride) {
    uint4 v = in4[i];
    count_word(local, v.x);
    count_word(local, v.y);
    count_word(local, v.z);
    count_word(local, v.w);
  }
  for (size_t j = nvec * 16 + tid; j < n; j += stride) {
    atomicAdd(&local[in[j]], 1u);
  }
  __syncthreads();

  for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
    if (local[b]) atomicAdd(&bins[b], local[b]);
  }
}

inline void launch_hist_shared_vec(const uint8_t* d_in,
                                   unsigned* d_bins, size_t n,
                                   cudaStream_t s = 0) {
  hist_shared_vec<<<sm_count() * 8, 256, 0, s>>>(
      d_in, d_bins, n);
}

// Version 3: warp-aggregated updates, scalar loads.
__global__ void hist_warp_agg(const uint8_t* __restrict__ in,
                              unsigned* __restrict__ bins,
                              size_t n) {
  __shared__ unsigned local[kBins];
  for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
    local[b] = 0;
  }
  __syncthreads();

  const unsigned lane = threadIdx.x & 31;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  // The loop bound depends only on blockIdx, so every lane of
  // a warp reaches __match_any_sync together.
  for (size_t base = (size_t)blockIdx.x * blockDim.x;
       base < n; base += stride) {
    const size_t idx = base + threadIdx.x;
    const bool valid = idx < n;
    const unsigned v = valid ? in[idx] : (unsigned)kBins;
    const unsigned peers = __match_any_sync(0xffffffffu, v);
    if (valid && lane == (unsigned)(__ffs((int)peers) - 1)) {
      atomicAdd(&local[v], (unsigned)__popc(peers));
    }
  }
  __syncthreads();

  for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
    if (local[b]) atomicAdd(&bins[b], local[b]);
  }
}

inline void launch_hist_warp_agg(const uint8_t* d_in,
                                 unsigned* d_bins, size_t n,
                                 cudaStream_t s = 0) {
  hist_warp_agg<<<sm_count() * 8, 256, 0, s>>>(
      d_in, d_bins, n);
}
