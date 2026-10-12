#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace ch04 {

inline std::vector<uint8_t> make_uniform(size_t n,
                                         unsigned seed) {
  std::mt19937 gen(seed);
  std::vector<uint8_t> v(n);
  for (auto& x : v) x = static_cast<uint8_t>(gen() >> 24);
  return v;
}

// Ninety percent of pixels share one value, like a flat sky.
inline std::vector<uint8_t> make_skewed(size_t n,
                                        unsigned seed) {
  std::mt19937 gen(seed);
  std::vector<uint8_t> v(n);
  for (auto& x : v) {
    uint32_t r = gen();
    x = (r % 100 < 90) ? 128 : static_cast<uint8_t>(r >> 24);
  }
  return v;
}

inline std::vector<unsigned> cpu_hist(
    const std::vector<uint8_t>& v) {
  std::vector<unsigned> h(256, 0);
  for (uint8_t x : v) ++h[x];
  return h;
}

}  // namespace ch04
