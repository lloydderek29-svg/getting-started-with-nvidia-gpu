#include <cstdio>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

__global__ void write_ids(int* out) {
  out[threadIdx.x] = threadIdx.x;
}

static void print_version(const char* label, int v) {
  std::printf("%-16s: %d.%d\n", label,
              v / 1000, (v % 1000) / 10);
}

int main() {
  int driver = 0, runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  print_version("Driver supports", driver);
  print_version("Runtime", runtime);

  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(int)));
  write_ids<<<1, 32>>>(d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_out[32];
  CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));

  for (int i = 0; i < 32; ++i) {
    if (h_out[i] != i) {
      std::fprintf(stderr, "Kernel result mismatch\n");
      return 1;
    }
  }
  std::printf("Kernel launch and copy back: OK\n");
  return 0;
}
