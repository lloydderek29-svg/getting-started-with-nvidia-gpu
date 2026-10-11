#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"
#include "gpubook/timer.cuh"

static double gbps(void* dst, const void* src, size_t bytes,
                   cudaMemcpyKind kind, int reps) {
  gpubook::GpuTimer timer;
  CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));  // warm-up
  timer.start();
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
  }
  float ms = timer.stop();
  return (double)bytes * reps / (ms * 1e-3) / 1e9;
}

int main() {
  const size_t bytes = 256ull << 20;
  const int reps = 10;

  void* d_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, bytes));

  void* pageable = std::malloc(bytes);
  void* pinned = nullptr;
  CUDA_CHECK(cudaMallocHost(&pinned, bytes));
  std::memset(pageable, 1, bytes);
  std::memset(pinned, 1, bytes);

  std::printf("Host to device, pageable: %6.1f GB/s\n",
      gbps(d_buf, pageable, bytes,
           cudaMemcpyHostToDevice, reps));
  std::printf("Host to device, pinned  : %6.1f GB/s\n",
      gbps(d_buf, pinned, bytes,
           cudaMemcpyHostToDevice, reps));
  std::printf("Device to host, pageable: %6.1f GB/s\n",
      gbps(pageable, d_buf, bytes,
           cudaMemcpyDeviceToHost, reps));
  std::printf("Device to host, pinned  : %6.1f GB/s\n",
      gbps(pinned, d_buf, bytes,
           cudaMemcpyDeviceToHost, reps));

  CUDA_CHECK(cudaFreeHost(pinned));
  std::free(pageable);
  CUDA_CHECK(cudaFree(d_buf));
  return 0;
}
