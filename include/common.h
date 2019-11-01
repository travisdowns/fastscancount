#ifndef COMMON_H_
#define COMMON_H_

#include <assert.h>
#include <inttypes.h>

#include <vector>

// #define DEBUGB 1

#ifdef DEBUGB
#define DBG(...) __VA_ARGS__
#else
#define DBG(...)
#endif

using data_array = std::vector<uint32_t>;
using all_data = std::vector<data_array>;
using data_ptrs = std::vector<const data_array *>;

/* calculates p / q, rounded up, both p and q must be non-negative */
template <typename T>
inline T div_up(T p, T q) {
  assert(p >= 0);
  assert(q >= 0);
  return (p + q - 1) / q;
}

constexpr size_t lg2(size_t n) {
    assert(n > 0);
    if (n > 1) {
        return 1 + lg2(n / 2);
    } else {
        return 0;
    }
}

constexpr size_t lg2_up(size_t n) {
  assert(n > 0);
  return n == 1 ? 0 : lg2(n - 1) + 1;
}

static inline uint32_t get_largest(const std::vector<std::vector<uint32_t>>& data) {
  uint32_t largest = 0;
  for (auto& v : data) {
    if (!v.empty())
      largest = std::max(largest, v.back());
  }
  return largest;
}

static inline uint32_t get_smallest_max(const std::vector<std::vector<uint32_t>>& data) {
  uint32_t smallest = -1;
  for (auto& v : data) {
    if (!v.empty())
      smallest = std::min(smallest, v.back());
  }
  return smallest;
}

template <typename T>
struct minispan {
  T* begin;
  size_t size_;

  const T& operator[](size_t i) const {
    assert(i < size());
    return begin[i];
  }

  size_t size() const { return size_; }

  template <typename C>
  static minispan<T> from(C& c) {
    return { c.data(), c.size() };
  }
};

/**
 * Return the alignmetn of the given pointer, i.e,. the largest power
 * of which the address is a multiple.
 */
inline size_t get_alignment(const void *p) {
  return (size_t)((1UL << __builtin_ctzl((uintptr_t)p)));
}


#endif // #ifndef COMMON_H_
