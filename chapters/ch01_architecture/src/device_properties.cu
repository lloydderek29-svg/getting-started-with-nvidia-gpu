#include <cstdio>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

static int attr(cudaDeviceAttr a, int dev) {
  int v = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
  return v;
}

int main() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  std::printf("CUDA devices found: %d\n", count);

  for (int dev = 0; dev < count; ++dev) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    const double gib = 1024.0 * 1024.0 * 1024.0;

    std::printf("\nDevice %d: %s\n", dev, p.name);
    std::printf("  Compute capability   : %d.%d\n",
                p.major, p.minor);
    std::printf("  Streaming MPs (SMs)  : %d\n",
                p.multiProcessorCount);
    std::printf("  Global memory        : %.2f GiB\n",
                p.totalGlobalMem / gib);
    std::printf("  L2 cache             : %.1f MiB\n",
                p.l2CacheSize / (1024.0 * 1024.0));
    std::printf("  Memory bus width     : %d bit\n",
                p.memoryBusWidth);
    std::printf("  Memory clock         : %d MHz\n",
        attr(cudaDevAttrMemoryClockRate, dev) / 1000);
    std::printf("  SM clock (max)       : %d MHz\n",
        attr(cudaDevAttrClockRate, dev) / 1000);
    std::printf("  Warp size            : %d\n",
                p.warpSize);
    std::printf("  Max threads / block  : %d\n",
                p.maxThreadsPerBlock);
    std::printf("  Max threads / SM     : %d\n",
                p.maxThreadsPerMultiProcessor);
    std::printf("  Registers / SM       : %d\n",
                p.regsPerMultiprocessor);
    std::printf("  Shared mem / block   : %zu KiB\n",
                p.sharedMemPerBlock / 1024);
    std::printf("  Shared mem / SM      : %zu KiB\n",
                p.sharedMemPerMultiprocessor / 1024);
    std::printf("  Async copy engines   : %d\n",
                p.asyncEngineCount);
    std::printf("  Concurrent kernels   : %s\n",
                p.concurrentKernels ? "yes" : "no");
    std::printf("  Managed memory       : %s\n",
                p.managedMemory ? "yes" : "no");
    std::printf("  ECC enabled          : %s\n",
                p.ECCEnabled ? "yes" : "no");
    std::printf("  Integrated GPU       : %s\n",
                p.integrated ? "yes" : "no");
  }
  return 0;
}
