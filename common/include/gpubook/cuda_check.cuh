// #pragma once
// #include <cstdio>
// #include <cstdlib>
// #include <cuda_runtime.h>

// #define CUDA_CHECK(call)                                  \
//   do {                                                    \
//     cudaError_t err__ = (call);                           \
//     if (err__ != cudaSuccess) {                           \
//       std::fprintf(stderr, "CUDA error %s at %s:%d\n",    \
//                    cudaGetErrorName(err__),               \
//                    __FILE__, __LINE__);                   \
//       std::fprintf(stderr, "  %s\n",                      \
//                    cudaGetErrorString(err__));            \
//       std::exit(EXIT_FAILURE);                            \
//     }                                                     \
//   } while (0)
#pragma once
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace gpubook {

class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t code, const char* file, int line)
      : std::runtime_error(build(code, file, line)),
        code_(code) {}
  cudaError_t code() const { return code_; }

 private:
  static std::string build(cudaError_t c, const char* f,
                           int l) {
    return std::string(cudaGetErrorName(c)) + ": " +
           cudaGetErrorString(c) + " (" + f + ":" +
           std::to_string(l) + ")";
  }
  cudaError_t code_;
};

}  // namespace gpubook

#define CUDA_CHECK(call)                                  \
  do {                                                    \
    cudaError_t err__ = (call);                           \
    if (err__ != cudaSuccess) {                           \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",    \
                   cudaGetErrorName(err__),               \
                   __FILE__, __LINE__);                   \
      std::fprintf(stderr, "  %s\n",                      \
                   cudaGetErrorString(err__));            \
      std::exit(EXIT_FAILURE);                            \
    }                                                     \
  } while (0)

#define CUDA_CHECK_THROW(call)                            \
  do {                                                    \
    cudaError_t err__ = (call);                           \
    if (err__ != cudaSuccess) {                           \
      throw gpubook::CudaError(err__, __FILE__, __LINE__);\
    }                                                     \
  } while (0)

#ifdef GPUBOOK_DEBUG_SYNC
#define CUDA_CHECK_LAST()                                 \
  do {                                                    \
    CUDA_CHECK(cudaGetLastError());                       \
    CUDA_CHECK(cudaDeviceSynchronize());                  \
  } while (0)
#else
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
#endif
