#ifndef COMMON_H_
#define COMMON_H_

#include <inttypes.h>

/* calculates p / q, rounded up, both p and q must be non-negative */
template <typename T>
inline T div_up(T p, T q) {
  assert(p >= 0);
  assert(q >= 0);
  return (p + q - 1) / q;
}

uint32_t get_largest(const std::vector<std::vector<uint32_t>>& data) {
  uint32_t largest = 0;
  for (auto& v : data) {
    if (!v.empty())
      largest = std::max(largest, v.back());
  }
  return largest;
}

uint32_t get_smallest_max(const std::vector<std::vector<uint32_t>>& data) {
  uint32_t smallest = -1;
  for (auto& v : data) {
    if (!v.empty())
      smallest = std::min(smallest, v.back());
  }
  return smallest;
}

#endif // #ifndef COMMON_H_
