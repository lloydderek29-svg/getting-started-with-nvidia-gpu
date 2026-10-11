#pragma once
#include <cstddef>

namespace ch03 {

// Sum of n floats stored in device memory.
float gpu_sum(const float* d_in, size_t n);

// C = A * B, all row-major device pointers.
// A is MxK, B is KxN, C is MxN.
void gpu_matmul(const float* d_a, const float* d_b,
                float* d_c, int m, int n, int k);

}  // namespace ch03
