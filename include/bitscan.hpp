#ifndef BITSCAN_H_
#define BITSCAN_H_

#include "accum7.hpp"
#include "compressed-bitmap.hpp"
#include "common.h"
#include "hedley.h"

#include <stddef.h>
#include <inttypes.h>
#include <vector>
#include <bitset>

#include <immintrin.h>


namespace fastscancount {

template <typename T>
struct bitscan_all_aux {
  uint32_t largest; // the largest value found in any array
  std::vector<compressed_bitmap<T>> bitmaps;

  bitscan_all_aux(uint32_t largest) : largest{largest} {}

  size_t get_chunk_count() const {
      return div_up(largest + 1u, 512u);
  }
};


template <typename T>
HEDLEY_NEVER_INLINE
bitscan_all_aux<T> get_all_aux_bitscan(const all_data& data) {
  bitscan_all_aux<T> ret(get_largest(data));
  auto& bitmaps = ret.bitmaps;
  bitmaps.reserve(data.size());
  for (auto &d : data) {
    bitmaps.emplace_back(d, ret.largest);
    assert(bitmaps.back().control.size() == ret.get_chunk_count());
  }
  return ret;
}

template <typename T>
void bitscan_scalar(const data_ptrs &, std::vector<uint32_t> &out,
                    uint8_t threshold, const bitscan_all_aux<T>& aux_info,
                    const std::vector<uint32_t>& query)
{
    std::vector<uint8_t> counters(aux_info.largest + 1);
    for (auto did : query) {
        auto bitmap = aux_info.bitmaps.at(did);
        for (auto id : bitmap.indices()) {
            ++counters.at(id);
        }
    }

    for (size_t i = 0; i < counters.size(); i++) {
        if (counters.at(i) > threshold) {
            out.push_back(i);
        }
    }
}

template <typename U>
struct chunk_traits : traits_base<typename compressed_bitmap<U>::chunk_type, chunk_traits<U>> {
    using T = typename compressed_bitmap<U>::chunk_type;

    static T xor_(const T& l, const T& r) {
        return l ^ r;
    }

    static T and_(const T& l, const T& r) {
        return l & r;
    }

    static T or_(const T& l, const T& r) {
        return l | r;
    }

    static T not_(const T& v) {
        return ~v;
    }

    static bool test(const T& v, size_t idx) {
        return v[idx];
    }

    static size_t size() {
        return compressed_bitmap<U>::chunk_bits;
    }

    static bool zero(const T& v) {
        return v.none();
    }
};



/** basic SIMD algorithm, but fake */
template <typename T>
void bitscan_fake(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<T>& aux_info,
                const std::vector<uint32_t>& query)
{
    using btype = compressed_bitmap<T>;
    using chunk_type = typename btype::chunk_type;

    std::vector<accumulator<7, chunk_type, chunk_traits<T>>> accums;
    accums.resize(aux_info.get_chunk_count());

    for (auto did : query) {
        auto& bitmap = aux_info.bitmaps.at(did);
        auto chunks = bitmap.chunks();
        assert(chunks.size() == aux_info.get_chunk_count());
        for (size_t c = 0; c < chunks.size(); c++) {
            // printf("expanded had %zu bits\n", chunks[c].count());
            accums.at(c).accept(chunks.at(c));
        }
    }

    size_t offset = 0;
    for (auto& accum : accums) {
        auto sums = accum.get_sums();
        for (size_t s = 0; s < sums.size(); s++) {
            if (sums[s] > threshold) {
                out.push_back(offset + s);
            }
        }
        offset += btype::chunk_bits;
    }
}

template <typename T>
void bitscan_fake2(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<T>& aux_info,
                const std::vector<uint32_t>& query);

#ifdef __AVX512F__

inline fastbitset<512> to_bitset(__m512i v) {
    static_assert(sizeof(fastbitset<512>) * 8 == 512);
    fastbitset<512> ret;
    _mm512_storeu_si512(&ret, v);
    return ret;
}

template <typename T>
void bitscan_avx512(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<T>& aux_info,
                const std::vector<uint32_t>& query);

void bitscan_avx512_asm(const data_ptrs &, std::vector<uint32_t> &out,
        uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
        const std::vector<uint32_t>& query);

struct m512_traits {
    using T = __m512i;

    static T xor_(const T& l, const T& r) {
        return _mm512_xor_si512(l, r);
    }

    static T and_(const T& l, const T& r) {
        return _mm512_and_si512(l, r);
    }

    static T or_(const T& l, const T& r) {
        return _mm512_or_si512(l, r);
    }

    static T not_(const T& v) {
        return _mm512_ternarylogic_epi32(v, v, v, 0x55);
    }

    static bool test(const T& v, size_t idx) {
        auto bitset = to_bitset(v);
        return bitset.test(idx);
    }

    static size_t size() {
        return 512;
    }

    static bool zero(const T& v) {
        return _mm512_cmpneq_epu32_mask(v, _mm512_setzero_si512()) == 0;
    }

    struct cs {
        T carry, sum;
    };

    /**
     * vpternlogd cheatsheet:
     * a = 0xF0
     * b = 0xCC
     * c = 0xAA
     */

    /* aka full adder */
    static cs add3(T a, T b, T c) {
        return {
            _mm512_ternarylogic_epi32(a, b, c, 0xE8),  // carry
            _mm512_ternarylogic_epi32(a, b, c, 0x96)   // sum
        };
    }

    /* aka half adder */
    static cs add2(T a, T b) {
        return {and_(a, b), xor_(a, b)};
    }
};
#endif



} // namespace fastscancount


#endif
