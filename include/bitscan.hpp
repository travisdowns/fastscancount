#ifndef BITSCAN_H_
#define BITSCAN_H_

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

/** a fake __mm512i class */
struct mm512fake {
    std::bitset<512> bits;

    mm512fake() {
    }

    // explicit mm512fake(const void* addr) {

    // }
};



template <typename T>
struct default_traits {

    struct carry_sum {
        T carry;
        T sum;
    };


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

    /* aka half adder */
    static carry_sum add2(T a, T b) {
        return {and_(a, b), xor_(a, b)};
    }

    /* aka full adder */
    static carry_sum add3(T a, T b, T c) {
        auto xor01 = xor_(a, b);
        return {
            or_(and_(a, b), and_(c, xor01)), // carry
            xor_(xor01, c) };                // sum
    }



    static bool test(const T& v, size_t idx) {
        return v[idx];
    }
};

template <typename U>
struct chunk_traits {
    using T = typename compressed_bitmap<U>::chunk_type;

    static T xor_(const T& l, const T& r) {
        T ret(l);
        ret ^= r;
        return ret;
    }

    static T and_(const T& l, const T& r) {
        T ret(l);
        ret &= r;
        return ret;
    }

    static T or_(const T& l, const T& r) {
        T ret(l);
        ret |= r;
        return ret;
    }

    static T not_(const T& v) {
        T ret(v);
        ~ret;
        return ret;
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



/**
 * Accumulator that can accumulate per-bit sums up to B bits. When a
 * value reaches the max of 2^B, it saturates.
 */
template <size_t B, typename T, typename traits = default_traits<T>>
class accumulator {
    T bits[B + 1]; // the last element is set for saturation

public:
    static constexpr size_t max = ((size_t)1) << B;

    accumulator(size_t initial = 0) : bits{} {
        T ones = traits::not_(T{});
        while (initial--) {
            accept(ones);
        }
    }

    void accept(const T& addend) {
        T carry = addend;
        for (size_t bit = 0; bit < B; bit++) {
            T sum = traits::xor_(bits[bit], carry);
            carry = traits::and_(bits[bit], carry);
            bits[bit] = sum;
        }
        bits[B] = traits::or_(bits[B], carry);  // update saturation
    }

    /**
     * A vector with one element per vertical counter containing its sum.
     *
     * This method is very slow and is intended primarily for writing tests.
     */
    std::vector<size_t> get_sums() {
        std::vector<size_t> ret(traits::size());
        for (size_t i = 0; i < traits::size(); i++) {
            size_t multiplier = 1;
            for (size_t bit = 0; bit < B; bit++) {
                ret[i] += traits::test(bits[bit], i) * multiplier;
                multiplier *= 2;
            }
            if (traits::test(bits[B], i)) {
                ret[i] = max; // saturation
            }
        }
        return ret;
    }

    T get_saturated() {
        return bits[B];
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

/** basic SIMD algorithm, but fake */
template <typename T>
void bitscan_avx512(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<T>& aux_info,
                const std::vector<uint32_t>& query);

#ifdef __AVX512F__

inline boost::dynamic_bitset<> to_bitset(__m512i v) {
    using block_type = boost::dynamic_bitset<>::block_type;
    block_type blocks[sizeof(__m512i) / sizeof(block_type)];
    static_assert(sizeof(blocks) == sizeof(v), "huh");
    _mm512_storeu_si512(&blocks, v);
    return boost::dynamic_bitset<>(blocks, blocks + sizeof(blocks) / sizeof(blocks[0]));
}

struct m512_traits : default_traits<__m512i> {
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
        return bitset[idx];
    }

    static size_t size() {
        return 512;
    }

    static bool zero(const T& v) {
        return to_bitset(v).none();
    }
};
#endif



} // namespace fastscancount


#endif
