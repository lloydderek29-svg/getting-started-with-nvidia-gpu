#pragma once
#include <cuda_runtime.h>

namespace gpubook {

inline cudaError_t prefetch_to_device(
    const void* ptr, size_t bytes, int device,
    cudaStream_t stream = 0) {
#if CUDART_VERSION >= 13000
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = device;
  return cudaMemPrefetchAsync(ptr, bytes, loc, 0, stream);
#else
  return cudaMemPrefetchAsync(ptr, bytes, device, stream);
#endif
}

}  // namespace gpubook
