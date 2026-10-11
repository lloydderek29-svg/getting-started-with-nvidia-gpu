#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "gpubook/cuda_check.cuh"
#include "gpubook/timer.cuh"

__inline__ __device__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__global__ void reduce_sum(const float* __restrict__ in,
                           float* __restrict__ out,
                           size_t n) {
  __shared__ float warp_sums[32];

  float sum = 0.0f;
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    sum += in[i];
  }

  sum = warp_reduce_sum(sum);
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();

  const int num_warps = blockDim.x / 32;
  sum = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum(sum);
  if (threadIdx.x == 0) atomicAdd(out, sum);
}

int main() {
  const size_t n = 1 << 26;
  std::vector<float> h(n);
  double ref = 0.0;
  for (size_t i = 0; i < n; ++i) {
    h[i] = (float)((i % 7) - 3) * 0.25f;
    ref += h[i];
  }

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  int dev = 0, sms = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &sms, cudaDevAttrMultiProcessorCount, dev));
  const int block = 256;
  const int grid = sms * 8;

  gpubook::GpuTimer timer;
  float result = 0.0f;

  // Custom kernel
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  reduce_sum<<<grid, block>>>(d_in, d_out, n);  // warm-up
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  timer.start();
  reduce_sum<<<grid, block>>>(d_in, d_out, n);
  float ms_custom = timer.stop();
  CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::printf("Custom kernel: %.3f ms, sum=%.2f "
              "(reference %.2f)\n", ms_custom, result, ref);

  // CUB reference
  void* d_temp = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
  cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
  timer.start();
  cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
  float ms_cub = timer.stop();
  CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::printf("CUB          : %.3f ms, sum=%.2f\n",
              ms_cub, result);

  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
