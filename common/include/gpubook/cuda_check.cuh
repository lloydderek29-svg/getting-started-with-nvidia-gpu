#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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
