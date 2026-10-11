#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_padded(float* out, const float* in,
                                 int width, int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < width && (y + j) < height) {
      tile[threadIdx.y + j][threadIdx.x] =
          in[(y + j) * width + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < height && (y + j) < width) {
      out[(y + j) * height + x] =
          tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

int main() {
  const int width = 1000, height = 700;
  const size_t n = (size_t)width * height;
  std::vector<float> h_in(n), h_out(n);
  for (size_t i = 0; i < n; ++i) h_in[i] = (float)i;

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                        n * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block(TILE_DIM, BLOCK_ROWS);
  dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
            (height + TILE_DIM - 1) / TILE_DIM);
  transpose_padded<<<grid, block>>>(d_out, d_in,
                                    width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                        n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      if (h_out[(size_t)c * height + r] !=
          h_in[(size_t)r * width + c]) {
        std::printf("Mismatch at (%d,%d)\n", r, c);
        return 1;
      }
    }
  }
  std::printf("Transpose %dx%d: OK\n", width, height);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
