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

  int runtime = 0, driver = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));

  for (int dev = 0; dev < count; ++dev) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int bus = attr(cudaDevAttrGlobalMemoryBusWidth, dev);
    int mclk = attr(cudaDevAttrMemoryClockRate, dev);
    double peak = 2.0 * mclk * 1e3 * (bus / 8.0) / 1e9;

    std::printf("{\"index\":%d,", dev);
    std::printf("\"name\":\"%s\",", p.name);
    std::printf("\"cc\":\"%d.%d\",", p.major, p.minor);
    std::printf("\"sm_count\":%d,", p.multiProcessorCount);
    std::printf("\"max_threads_per_sm\":%d,",
                p.maxThreadsPerMultiProcessor);
    std::printf("\"regs_per_sm\":%d,",
                p.regsPerMultiprocessor);
    std::printf("\"smem_per_sm\":%zu,",
                p.sharedMemPerMultiprocessor);
    std::printf("\"l2_bytes\":%d,", p.l2CacheSize);
    std::printf("\"bus_bits\":%d,", bus);
    std::printf("\"peak_bw_gbs\":%.1f,", peak);
    std::printf("\"runtime_version\":%d,", runtime);
    std::printf("\"driver_version\":%d}\n", driver);
  }
  return 0;
}
