#include <cstdio>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

__global__ void scale(float* data, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= a;
}

int main() {
  int min_grid = 0, best_block = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &min_grid, &best_block, scale, 0, 0));
  std::printf("Suggested block size : %d\n", best_block);
  std::printf("Min grid for full use: %d\n", min_grid);

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  const int block = 256;
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, scale, block, 0));

  double active = blocks_per_sm * block / 32.0;
  double max_warps = prop.maxThreadsPerMultiProcessor / 32.0;
  std::printf("Blocks per SM (256)  : %d\n", blocks_per_sm);
  std::printf("Theoretical occupancy: %.0f %%\n",
              100.0 * active / max_warps);
  return 0;
}
