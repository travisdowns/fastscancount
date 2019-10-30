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
#include <memory>
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


constexpr size_t cache_size = 40000 / 64 * 64;
constexpr size_t COUNTER_OFFSET = 64;
constexpr size_t counters_size = COUNTER_OFFSET + cache_size * 2;

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
 *
 * The object is parameterized on T, the type of elements pointed to
 * by the rewritten data.
 */
template <typename T>
struct aux_chunk_t {
  aux_chunk_t() {} // leave everything uninitialized to avoid zeroing everything during vector creation
  aux_chunk_t(const T* start_ptr, uint32_t iter_count, uint32_t overshoot)
      : start_ptr{start_ptr}, iter_count{iter_count}, overshoot{overshoot} {}

  /* start of data array */
  const T* start_ptr;
  /* how many iterations are needed for this array at the current unroll factor */
  uint32_t iter_count;
  /* the overshoot for this chunk based on current unroll */
  uint32_t overshoot;
};

template <typename T>
struct aux_view_t {
  const T* data;
  minispan<const aux_chunk_t<T>> chunks;
};

/**
 * Auxilliary per-array data for avx2b algo.
 */
template <typename T>
struct avx2b_aux_t {
  using aux_chunk = aux_chunk_t<T>;
  using aux_view = aux_view_t<T>;

  uint32_t largest;

  /* pointer to the re-written data */
  std::unique_ptr<T[]> data;

  /* a vector of how many times the counter increment loop has to run for each cache_size chunk */
  std::vector<aux_chunk> chunks;

  avx2b_aux_t(const data_array& array, uint32_t global_largest) {

    assert(!array.empty());  // some bugs with empty arrays
    this->largest = array.back();

    size_t pos = 0;
    size_t chunk_count = div_up(global_largest + 1, (uint32_t)cache_size);
    DBG(printf("chunk_count: %zu\n", chunk_count));
    size_t c = 0;

    // we rewrite the data into this vector
    std::vector<T> rewritten;
    rewritten.reserve(global_largest);

    // keep track of how many filer elements we write
    size_t filler_total = 0;

    // and keep track of what happened to each chunk here
    struct rw_meta {
      // index into rewritten
      size_t rw_index;
      uint32_t iter_count;
      uint32_t overshoot;
    };
    std::vector<rw_meta> meta;

    for (uint32_t rstart = 0; rstart <= global_largest; rstart += cache_size, c++) {
      // striding by the unroll factor, look for the point in the array where
      // the current element falls outside of the range, this is where we transition
      // in the branch free loop
      uint32_t rend = rstart + cache_size; // exclusive
      size_t spos = pos;
      while (pos < array.size() && array.at(pos) < rend) {
        pos += unroll;
      }

      if (pos > array.size()) {
        pos = array.size();
      }

      // TODO: do we need limit of >= 1 loops for branchy?
      size_t loops = std::max(1ul, div_up(pos - spos, unroll));  // round up

      // I heard you like asserts
      assert(spos < pos || (spos == pos && pos == array.size()));
      if (spos < array.size()) {
        assert(spos % unroll == 0);
        assert(array.at(spos) >= rstart);
        if (pos < array.size()) {
          assert(pos % unroll == 0);
          assert(array.at(pos) >= rend);
          assert(spos + loops * unroll == pos);
          assert((pos - spos) % unroll == 0);
        }
      }
      // unless we hit the single (forced) chunk case, the last block should be in the range of this chunk
      assert(array.at(pos - unroll) < rend || pos - unroll == spos);

      assert(loops < std::numeric_limits<uint32_t>::max());

      uint32_t last = array.at(pos - 1); // check the last element in the last block for this chunk, this defines the overshoot
      assert(last > rstart || spos == pos);
      uint32_t overshoot = last < rend ? 0 : last - rend + 1;
      assert(overshoot < cache_size); // this could happen for real data, need to fix "extended overshoot" to solve it

      size_t rw_index = rewritten.size();
      for (size_t i = spos; i < pos; i++) {
        uint32_t val = array.at(i) + COUNTER_OFFSET;
        assert(val >= rstart);

        // rebase the value to be relative to rstart (i.e., in the range [0, cache_size] + COUNTER_OFFSET)
        val -= rstart;
        assert(val <= std::numeric_limits<T>::max());
        assert(val == (T)val);
        assert(val < counters_size);

        rewritten.push_back(val);
      }
      ssize_t filler_count = loops * unroll - (pos - spos); // amount extra we have to write
      filler_total += filler_count;
      assert(filler_count >= 0);
      assert(filler_count == 0 || pos == array.size());

      // fill out the block with zeros for any filler elements - because of COUNTER_OFFSET
      // zeros write to an ignored part of the array and hence serve as no ops
      rewritten.insert(rewritten.end(), filler_count, 0);
      // the amount written must be equal to the amount the counter code
      // will read
      size_t written_count = rewritten.size() - rw_index;
      assert(loops * unroll == written_count);
      meta.push_back({rw_index, (uint32_t)loops, overshoot});
      DBG(printf("AUX - chunk: %zu a[%zu]: %u to a[%zu] : %u iters: %zu "
          "wreal: %zu wfill: %zu oshoot: %5du rstart %du\n",
          c, spos, array[spos], pos - 1, array[pos - 1], loops,
          written_count - filler_count, filler_count, overshoot, rstart);)
    }

    assert(chunk_count == c);
    assert(meta.size() == chunk_count);

    data.reset(new T[rewritten.size()]);
    std::copy(rewritten.begin(), rewritten.end(), data.get());

    chunks.reserve(chunk_count);

    for (auto& m : meta) {
      auto sptr = data.get() + m.rw_index;
      // DBG(printf("sptr: %p send: %p\n", sptr, sptr + unroll * m.iter_count);)
      chunks.push_back({sptr, m.iter_count, m.overshoot});
    }

    DBG(printf("Filler %%%7.4f\n", 100.0 * filler_total / rewritten.size()));
  }

  /**
   * Returns a "view" of this aux object. A view points to the
   * same data and chunk vector, but is non-owning. A view is valid
   * only as long as the object it was created from is valid (and
   * the corresponding fields are not modified).
   */
  aux_view get_view() const {
    return { data.get(), minispan<const aux_chunk>::from(chunks) };
  }
};

template <typename T>
struct all_aux_t {
  uint32_t largest; // the largest value found in any array
  std::vector<avx2b_aux_t<T>> aux_data;

  all_aux_t(uint32_t largest) : largest{largest} {}
};

/**
 * Auxillary data specific to a query, calculated dynamically based
 * on the aux data from the input arrays.
 */
template <typename T>
struct dynamic_aux {
  using aux_chunk = aux_chunk_t<T>;
  using aux_view = aux_view_t<T>;

  uint32_t largest;

  std::vector<std::vector<aux_chunk>> aux;
  std::vector<uint32_t> max_overshoot;

  HEDLEY_NEVER_INLINE
  dynamic_aux(const implb::all_aux_t<T>& all_aux_info, const std::vector<uint32_t>& data_indexes) {

      /* extract the relevant aux_info arrays based on the given data_indexes */
    std::vector<aux_view> views;
    uint32_t largest = 0;
    views.reserve(data_indexes.size());
    for (auto i : data_indexes) {
      assert(i < all_aux_info.aux_data.size());
      auto& aux_data = all_aux_info.aux_data[i];
      views.push_back(aux_data.get_view());
      largest = largest >= aux_data.largest ? largest : aux_data.largest;
    }
    this->largest = largest;

    size_t dsize = views.size();
    size_t chunks_needed = div_up(largest + 1, (uint32_t)cache_size);
    DBG(printf("chunks_needed: %zu\n", chunks_needed);)

    size_t pfdistance = std::min((size_t)30, dsize);

    for (auto& v : views) {
      assert(chunks_needed <= v.chunks.size());
    }

    for (size_t chunk = 0; chunk < chunks_needed; chunk++) {
      for (size_t i = 0; i < pfdistance; i++) {
        _mm_prefetch(&views[i].chunks[chunk], _MM_HINT_T0);
      }

      aux.emplace_back(dsize + 1);

      auto& thisaux = aux.back();
      // auto& ptr = start_ptr.back();
      uint32_t maxo = 0;

      for (size_t i = 0; i < dsize - pfdistance; ++i) {
        assert(chunk < views[i].chunks.size());
        auto info = views[i].chunks[chunk];
        _mm_prefetch(&views[i + pfdistance].chunks[chunk], _MM_HINT_T0);
        thisaux[i] = info;
        maxo = maxo > info.overshoot ? maxo : info.overshoot;
      }

      for (size_t i = dsize - pfdistance; i < dsize; ++i) {
        assert(chunk < views[i].chunks.size());
        auto info = views[i].chunks[chunk];
        thisaux[i] = info;
        maxo = maxo > info.overshoot ? maxo : info.overshoot;
      }

      thisaux[dsize] = thisaux[dsize - 1];
      DBG(printf("maxo: %du\n", maxo);)
      max_overshoot.push_back((maxo + 31) & -32);  // round up overshoot so the memset(0) is aligned
    }
  }
};

template <typename T>
HEDLEY_NEVER_INLINE
all_aux_t<T> get_all_aux(const all_data& data) {
  all_aux_t<T> ret{ get_largest(data) };
  auto& aux_array = ret.aux_data;
  aux_array.reserve(data.size());
  for (auto &d : data) {
    aux_array.emplace_back(d, ret.largest);
  }
  for (auto& aux : aux_array) {
    // DBG(printf("chunks.size(): %zu\n", aux.chunks.size()));
    assert(aux.chunks.size() == aux_array.front().chunks.size());
  }
  return ret;
}

/**
 * A version which gets the normal aux data on a per-input array
 * basis, but then totally cheats and copies all the data such that
 * it is linear when accessed by the counting algorithm.
 */
template <typename T>
HEDLEY_NEVER_INLINE
all_aux_t<T> get_all_aux_reordered(const all_data& data) {
  all_aux_t<T> all = get_all_aux<T>(data);

  std::vector<uint32_t> contiguous;
  contiguous.reserve(data.size() * data.front().size());
  for (uint32_t rstart = 0, chunk = 0; rstart <= all.largest; rstart += cache_size, chunk++) {
    for (auto& aux : all.aux_data) {
      auto& chunkaux = aux.chunks.at(chunk);
      contiguous.insert(contiguous.end(), chunkaux.start_ptr, chunkaux.start_ptr + chunkaux.iter_count * unroll);
    }
  }

  // just leak this for now
  uint32_t* contigarray = new uint32_t[contiguous.size()];
  std::copy(contiguous.begin(), contiguous.end(), contigarray);

  uint32_t* cur = contigarray;
  for (uint32_t rstart = 0, chunk = 0; rstart <= all.largest; rstart += cache_size, chunk++) {
    for (auto& aux : all.aux_data) {
      auto& chunkaux = aux.chunks.at(chunk);
      chunkaux.start_ptr = cur;
      cur += chunkaux.iter_count * unroll;
    }
  }

  return all;
}




} // implb namespace

#define counter_base (counters + COUNTER_OFFSET)
alignas(64) uint8_t counters[counters_size];

template <typename T>
using kernel_fn = void (const implb::aux_chunk_t<T>* aux_ptr,
                        const implb::aux_chunk_t<T>* aux_end,
                        uint32_t range_start);

using kernel_fn32 = kernel_fn<uint32_t>;
using kernel_fn16 = kernel_fn<uint16_t>;

extern "C" kernel_fn32 record_hits_asm_branchy32;
extern "C" kernel_fn16 record_hits_asm_branchy16;
extern "C" kernel_fn16 record_hits_asm_branchyB;
extern "C" kernel_fn32 record_hits_asm_branchless32;
extern "C" kernel_fn16 record_hits_asm_branchless16;


template <typename T>
HEDLEY_NEVER_INLINE
void record_hits_c(const implb::aux_chunk_t<T>* aux_ptr,
                   const implb::aux_chunk_t<T>* aux_end,
                   uint32_t range_start) {
  using aux_chunk = implb::aux_chunk_t<T>;

#ifndef NDEBUG
  const aux_chunk* aux_start = aux_ptr;
#endif
  const T* eptr = aux_ptr->start_ptr;
  assert(eptr);
  size_t iters_left = aux_ptr->iter_count;
  assert(iters_left > 0);

  for (;;) {
    for (int i = 0; i < unroll; i++) {
      T e = *eptr++;
      assert(e >= COUNTER_OFFSET || e == 0);
      assert(e < sizeof(counters));
      ++counters[e];
    }

    if (HEDLEY_LIKELY(--iters_left)) {
      continue;
    }

    // move to the next data array
    aux_ptr++;        // get next array pointer
    eptr = aux_ptr->start_ptr;  // update element ptr
    iters_left = aux_ptr->iter_count;

#ifndef NDEBUG
      if (aux_ptr != aux_end) {
        auto dsize = aux_end - aux_start;
        auto didx = aux_ptr - aux_start;
        assert(didx < dsize);
        DBG(printf("switching to index %zu (%zu loops, %u overshoot)\n",
            didx, iters_left, aux_ptr->overshoot);)
      }
#endif
    if (aux_ptr == aux_end) {
      break;
    }
  }
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
 *
 * @param data_indexes the list of indexes of posting arrays for this query
 */
template <typename T, kernel_fn<T> K>
void fastscancount_avx2b(const implb::data_ptrs &, std::vector<uint32_t> &out,
                         uint8_t threshold, const implb::all_aux_t<T>& all_aux_info,
                         const std::vector<uint32_t>& data_indexes) {

  using namespace implb;
  using aux_chunk = aux_chunk_t<T>;

  out.clear();
  const size_t dsize = data_indexes.size();

  dynamic_aux dyn_aux(all_aux_info, data_indexes);

  size_t chunk = 0;
  //memset(cdata, 0, 2 * cache_size * sizeof(counters[0]));
  memzero(counters, sizeof(counters));
  for (uint32_t range_start = 0, chunk = 0; range_start <= dyn_aux.largest; range_start += cache_size, chunk++) {
    uint32_t range_end = range_start + cache_size;

    assert(dyn_aux.aux.at(chunk).size() == dsize + 1);
    const aux_chunk* aux_ptr = dyn_aux.aux.at(chunk).data(), *aux_end = aux_ptr + dsize;

    DBG(printf("chunk %du range_start: %du end: %du iters_left %u first %u\n",
        chunk, range_start, range_end, aux_ptr->iter_count, *aux_ptr->start_ptr);)

    K(aux_ptr, aux_end, range_start);

    populate_hits_avx(counter_base, cache_size, threshold, range_start, out);

    uint32_t overshoot = dyn_aux.max_overshoot[chunk];
    //printf("overshoot: %u\n", overshoot);
    assert(overshoot < cache_size);
    // memcpy(counters, counters + cache_size, overshoot);
    copymem(counter_base, counter_base + cache_size, overshoot);
    // memset(counters + overshoot, 0, cache_size);
    memzero<cache_size>(counter_base + overshoot);
  }
}

template <kernel_fn32 K>
const auto fastscancount_avx2b32 = fastscancount_avx2b<uint32_t, K>;

} // namespace fastscancount
#endif
