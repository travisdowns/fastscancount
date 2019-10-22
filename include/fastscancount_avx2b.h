#ifndef FASTSCANCOUNT_AVX2B_H
#define FASTSCANCOUNT_AVX2B_H

// this code expects an x64 processor with AVX2

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <assert.h>

#include <stdio.h>

#include "hedley.h"
#include "common.h"

// #define DEBUGB 1

#ifdef DEBUGB
#define DBG(...) __VA_ARGS__
#else
#define DBG(...)
#endif

template <typename T>
void findM(T& t, const char *name) {
  if (std::find(t.begin(), t.end(), 19999213) != t.end()) {
    printf("DIDX found 19999213 in %s\n", name);
  } else {
    printf("DIDX  no   19999213 in %s\n", name);
  }
}

namespace fastscancount {
// credit: implementation and design by Travis Downes

constexpr size_t cache_size = 40000;
// constexpr size_t cache_size = 40000;
static_assert(cache_size % 64 == 0, "should be a multiple of 64");

constexpr size_t unroll = 16;

namespace implb {


static inline size_t find_next_gt(uint8_t *array, const size_t size,
                                  const uint8_t threshold) {
  size_t vsize = size / 32;
  __m256i *varray = (__m256i *)array;
  __m256i *varray_end = varray + vsize;
  const __m256i comprand = _mm256_set1_epi8(threshold);
  int bits = 0;

  for (__m256i *p = varray; p != varray_end; p++) {
    __m256i v = _mm256_loadu_si256(p);
    __m256i cmp = _mm256_cmpgt_epi8(v, comprand);
    if ((bits = _mm256_movemask_epi8(cmp))) {
      return (p - varray) * 32 + __builtin_ctz(bits);
    }
  }

  // tail handling
  for (size_t i = vsize * 32; i < size; i++) {
    auto v = array[i];
    if (v > threshold)
      return i;
  }

  return SIZE_MAX;
}

static inline size_t find_next_gt2(uint8_t *array, const size_t size,
                                  const uint8_t threshold) {
  size_t vsize = size / 64;
  __m256i *varray = (__m256i *)array;
  __m256i *varray_end = varray + vsize * 2;
  const __m256i comprand = _mm256_set1_epi8(threshold);
  int bits = 0;

  for (__m256i *p = varray; p != varray_end; p += 2) {
    __m256i v0 = _mm256_loadu_si256(p);
    __m256i v1 = _mm256_loadu_si256(p + 1);
    __m256i cmp0 = _mm256_cmpgt_epi8(v0, comprand);
    __m256i cmp1 = _mm256_cmpgt_epi8(v1, comprand);
    if ((bits = _mm256_movemask_epi8(cmp0))) {
      return (p - varray) * 32 + __builtin_ctz(bits);
    }
    if ((bits = _mm256_movemask_epi8(cmp1))) {
      return (p - varray) * 32 + 32 + __builtin_ctz(bits);
    }
  }

  // tail handling
  for (size_t i = vsize * 64; i < size; i++) {
    auto v = array[i];
    if (v > threshold)
      return i;
  }

  return SIZE_MAX;
}

HEDLEY_NEVER_INLINE
void populate_hits_avx(uint8_t *array, size_t range,
                       size_t threshold, size_t start,
                       std::vector<uint32_t> &out) {
  while (true) {
    size_t next = find_next_gt2(array, range, (uint8_t)threshold);
    if (next == SIZE_MAX)
      break;
    uint32_t hit = start + next;
    out.push_back(hit);
    range -= (next + 1);
    array += (next + 1);
    start += (next + 1);
  }
}

using data_array = std::vector<uint32_t>;
using all_data = std::vector<data_array>;
using data_ptrs = std::vector<const data_array *>;

/**
 * Each chunk of data in an input array has an associated aux_chunk
 * object.
 */
struct aux_chunk {
  aux_chunk() {} // leave everything uninitialized to avoid zeroing everything during vector creation
  aux_chunk(const uint32_t* start_ptr, uint32_t iter_count, uint32_t overshoot)
      : start_ptr{start_ptr}, iter_count{iter_count}, overshoot{overshoot} {}

  /* start of data array */
  const uint32_t* start_ptr;
  /* how many iterations are needed for this array at the current unroll factor */
  uint32_t iter_count;
  /* the overshoot for this chunk based on current unroll */
  uint32_t overshoot;
};

/**
 * Auxilliary per-array data for avx2b algo.
 */
struct avx2b_aux {
  uint32_t largest;

  /* a vector of how many times the counter increment loop has to run for each cache_size chunk */
  std::vector<aux_chunk> chunks;

  avx2b_aux(const data_array& array) : largest{array.back()} {
    assert(array.size() % unroll == 0);
    assert(largest < MAX_ELEM);
    size_t pos = 0;
    chunks.reserve(largest / cache_size + 1);
    for (uint32_t rstart = 0; rstart <= largest; rstart += cache_size) {
      // striding by the unroll factor, look for the point in the array where
      // the current element falls outside of the range, this is where we transition
      // in the branch free loop
      uint32_t rend = rstart + cache_size; // exclusive
      size_t spos = pos;
      pos += unroll; // minimum of one iteration
      while (pos < array.size() && array.at(pos) < rend) {
        pos += unroll;
      }

      assert(pos <= array.size());

      // printf("LOOPS: %zu: ", pos - spos);
      size_t loops = (pos - spos) / unroll;
      // printf("loops: %zu\n", loops);
      if (loops == 0) {
        DBG(printf("zero loops - size: %zu spos: %zu pos %zu rstart %du\n", array.size(), spos, pos, rstart);)
        assert(false);
        // TODO remove loops == 0 case
      } else {
        assert(pos > spos);
        assert(pos >= unroll);
        assert(array.at(spos) >= rstart);
        assert(pos == array.size() || array.at(pos) >= rend);
        // unless we hit the single (forced) chunk case, the last block should be in the range of this chunk
        assert(array.at(pos - unroll) < rend || pos - unroll == spos);
        auto sptr = array.data() + spos;
        assert(spos + loops * unroll <= array.size());
        DBG(printf("sptr: %p send: %p\n", sptr, sptr + unroll * loops);)
        assert(loops < std::numeric_limits<uint32_t>::max());
        uint32_t last = array.at(pos - 1); // check the last element in the last block for this chunk, this defines the overshoot
        assert(last > rstart);
        uint32_t overshoot = last < rend ? 0 : last - rend + 1;
        assert(overshoot < cache_size); // this could happen for real data, need to fixed "extended overshoot" to solve it
        chunks.push_back({sptr, (uint32_t)loops, overshoot});
        DBG(printf("AUX - a[%zu]: %u to a[%zu] : %u iters: %zu oshoot: %5du rstart %du\n",
            spos, array[spos], pos - 1, array[pos - 1], loops, overshoot, rstart);)
      }

      if (pos == array.size()) {
        break;
      }
    }
  }
};

using all_aux = std::vector<avx2b_aux>;

/**
 * Auxillary data specific to a query, calculated dynamically based
 * on the aux data from the input arrays.
 */
struct dynamic_aux {
  uint32_t largest;

  // std::vector<std::vector<size_t>> iter_counts;
  // std::vector<std::vector<const uint32_t *>> start_ptr;
  std::vector<std::vector<aux_chunk>> aux;
  std::vector<uint32_t> max_overshoot;

  HEDLEY_NEVER_INLINE
  dynamic_aux(const all_aux& aux_array) {
    uint32_t largest = 0;
    for (auto& aux : aux_array) {
      largest = std::max(largest, aux.largest);
    }
    this->largest = largest;

    size_t dsize = aux_array.size();
    size_t chunks = (largest + cache_size - 1) / cache_size;
    DBG(printf("chunks: %zu\n", chunks);)

    size_t pfdistance = std::min((size_t)30, dsize);

    for (size_t chunk = 0; chunk < chunks; chunk++) {
      for (size_t i = 0; i < pfdistance; i++) {
        _mm_prefetch(&aux_array[i].chunks[chunk], _MM_HINT_T0);
      }

      aux.emplace_back(dsize + 1);

      auto& thisaux = aux.back();
      // auto& ptr = start_ptr.back();
      uint32_t maxo = 0;

      for (size_t i = 0; i < dsize - pfdistance; ++i) {
        assert(chunk < aux_array[i].chunks.size());
        auto info = aux_array[i].chunks[chunk];
        _mm_prefetch(&aux_array[i + pfdistance].chunks[chunk], _MM_HINT_T0);
        thisaux[i] = info;
        maxo = maxo > info.overshoot ? maxo : info.overshoot;
      }

      for (size_t i = dsize - pfdistance; i < dsize; ++i) {
        assert(chunk < aux_array[i].chunks.size());
        auto info = aux_array[i].chunks[chunk];
        thisaux[i] = info;
        maxo = maxo > info.overshoot ? maxo : info.overshoot;
      }

      thisaux[dsize] = thisaux[dsize - 1];
      DBG(printf("maxo: %du\n", maxo);)
      max_overshoot.push_back((maxo + 31) & -32);  // round up overshoot so the memset(0) is aligned
    }
  }
};

HEDLEY_NEVER_INLINE
all_aux get_all_aux(const all_data& data) {
  all_aux ret;
  ret.reserve(data.size());
  for (auto &d : data) {
    ret.emplace_back(d);
  }
  for (auto& aux : ret) {
    assert(aux.chunks.size() == ret.front().chunks.size());
  }
  return ret;
}

/**
 * A version which gets the normal aux data on a per-input array
 * basis, but then totally cheats and copies all the data such that
 * it is linear when accessed by the counting algorithm.
 */
HEDLEY_NEVER_INLINE
all_aux get_all_aux_reordered(const all_data& data) {
  all_aux all = get_all_aux(data);

  std::vector<uint32_t> contiguous;
  contiguous.reserve(data.size() * data.front().size());
  for (uint32_t rstart = 0, chunk = 0; rstart < MAX_ELEM; rstart += cache_size, chunk++) {
    for (auto& aux : all) {
      auto& chunkaux = aux.chunks.at(chunk);
      contiguous.insert(contiguous.end(), chunkaux.start_ptr, chunkaux.start_ptr + chunkaux.iter_count * unroll);
    }
  }

  // just leak this for now
  uint32_t* contigarray = new uint32_t[contiguous.size()];
  std::copy(contiguous.begin(), contiguous.end(), contigarray);

  uint32_t* cur = contigarray;
  for (uint32_t rstart = 0, chunk = 0; rstart < MAX_ELEM; rstart += cache_size, chunk++) {
    for (auto& aux : all) {
      auto& chunkaux = aux.chunks.at(chunk);
      chunkaux.start_ptr = cur;
      cur += chunkaux.iter_count * unroll;
    }
  }

  return all;
}




} // implb namespace

alignas(64) uint8_t counters[cache_size * 2];

using kernel_fn = void (const implb::aux_chunk* aux_ptr,
                        const implb::aux_chunk* aux_end,
                        uint32_t range_start,
                        const implb::data_ptrs &data);

extern "C" kernel_fn record_hits_asm_branchy;
extern "C" kernel_fn record_hits_asm_branchless;
kernel_fn record_hits_c;

HEDLEY_NEVER_INLINE
void record_hits_c(const implb::aux_chunk* aux_ptr,
                   const implb::aux_chunk* aux_end,
                   uint32_t range_start,
                   const implb::data_ptrs &data) {
  const uint32_t* eptr = aux_ptr->start_ptr;
  assert(eptr);
  size_t iters_left = aux_ptr->iter_count;
  assert(iters_left > 0);

  do {
    for (int i = 0; i < unroll; i++) {
      uint32_t e = *eptr++;
#ifdef DEBUGB
      if (e < start) {
        auto didx = dsize - (start_ptrs_end - start_ptrs);
        printf("start: %du e: %du iters_left: %zu\n", start, e, iters_left);
        const uint32_t *sptr_begin = data[didx]->data();
        const uint32_t *sptr_end = sptr_begin + data[didx]->size();
        printf("sptr_begin %p\n", sptr_begin);
        printf("sptr_end   %p\n", sptr_end);
        printf("eptr       %p\n", eptr - 1);
      }
#endif
      assert(e >= range_start);
      assert(e - range_start < cache_size * 2);
      counters[e - range_start]++;
    }

    iters_left--;

    if (iters_left == 0) {
      // move to the next data array
      aux_ptr++;        // get next array pointer
      if (aux_ptr == aux_end)
        break;
      eptr = aux_ptr->start_ptr;  // update element ptr
      iters_left = aux_ptr->iter_count;

#ifndef NDEBUG
      if (aux_ptr == aux_end) {
        auto dsize = data.size();
        auto didx = dsize - (aux_end - aux_ptr);
        assert(didx < dsize);
        const uint32_t *sptr_begin = data[didx]->data();
        const uint32_t *sptr_end = sptr_begin + data[didx]->size();
        DBG(printf("switching to index %zu (%zu elements left)\n", didx, sptr_end - eptr);)
        assert(eptr >= sptr_begin);
        assert(eptr < sptr_end);
        assert(iters_left > 0);
        DBG(printf("iters_left: %zu\n", iters_left);)
        DBG(fflush(stdout);)
      }
#endif

    }

  } while (true);
}


size_t get_alignment(const void *p) {
  return (size_t)((1UL << __builtin_ctzl((uintptr_t)p)));
}

int zeroint;

HEDLEY_NEVER_INLINE
void memzero(void *dest, size_t count) {
  assert(count > 32);
  // printf("mzalign");
  size_t chunks = count / 32;
  __m256i zero = _mm256_set1_epi32(zeroint);
  __m256i* dest256 = (__m256i*)dest;
  for (long i = 0; i < chunks; i++) {
    _mm256_storeu_si256(dest256 + i, zero);
  }
  memset(dest256 + chunks, 0, count - chunks * 32);
}

template <size_t count, size_t U = 4>
HEDLEY_NEVER_INLINE
void memzero(void *dest) {
  constexpr size_t chunk_size = U * 32;
  static_assert(count >= chunk_size, "count too small");
  // printf("mzalign: %zu\n", get_alignment(dest));
  constexpr size_t chunks = count / chunk_size;
  __m256i zero = _mm256_set1_epi32(zeroint);
  __m256i* dest256 = (__m256i*)dest;
  for (long i = 0; i < chunks; i++) {
    for (size_t j = 0; j < U; j++) {
        _mm256_storeu_si256(dest256 + i * U + j, zero);
    }
  }
  constexpr auto rest = count - chunks * chunk_size;
  constexpr auto rest_vecs = (rest + 31) / 32;
  static_assert(rest_vecs * 32 < count, "count too small");
  dest256 = (__m256i*)((char *)dest + count - rest_vecs * 32);
  for (size_t j = 0; j < rest_vecs; j++) {
      _mm256_storeu_si256(dest256 + j, zero);
  }
}

template <typename T>
HEDLEY_NEVER_INLINE
void copymem(T* HEDLEY_RESTRICT dest, const T* src, size_t count) {
  for (T* end = dest + count; dest != end; dest++, src++) {
    *dest = *src;
  }
}

/**
 * Parameterized on K, the kernel function which does the core counter increment loop.
 */
template <kernel_fn K>
void fastscancount_avx2b(const implb::data_ptrs &data, std::vector<uint32_t> &out,
                         uint8_t threshold, const implb::all_aux& aux_info) {

  using namespace implb;

  out.clear();
  const size_t dsize = data.size();

  dynamic_aux dyn_aux(aux_info);

  auto cdata = counters;
  size_t chunk = 0;
  //memset(cdata, 0, 2 * cache_size * sizeof(counters[0]));
  memzero(cdata, 2 * cache_size * sizeof(counters[0]));
  for (uint32_t range_start = 0, chunk = 0; range_start < dyn_aux.largest; range_start += cache_size, chunk++) {
    uint32_t range_end = range_start + cache_size;

    assert(dyn_aux.aux.at(chunk).size() == dsize + 1);
    const aux_chunk* aux_ptr = dyn_aux.aux.at(chunk).data(), * aux_end = aux_ptr + dsize;

    DBG(printf("chunk %du range_start: %du end: %du iters_left %zu first %du\n",
        chunk, range_start, range_end, iters_left, *eptr);)

    K(aux_ptr, aux_end, range_start, data);

    populate_hits_avx(counters, cache_size, threshold, range_start, out);

    uint32_t overshoot = dyn_aux.max_overshoot[chunk];
    //printf("overshoot: %u\n", overshoot);
    assert(overshoot < cache_size);
    // memcpy(counters, counters + cache_size, overshoot);
    copymem(counters, counters + cache_size, overshoot);
    // memset(counters + overshoot, 0, cache_size);
    memzero<cache_size>(counters + overshoot);
  }
}

} // namespace fastscancount
#endif
