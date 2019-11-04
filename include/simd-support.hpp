#ifndef SIMD_SUPPORT_H_
#define SIMD_SUPPORT_H_

#include <cassert>
#include <cinttypes>
#include <immintrin.h>
#include <iosfwd>
#include <vector>

// #include "dbg.h"

extern uint8_t g_pack_left_table_uint8_tx3[256 * 3 + 1];

/** epi32 fill-in based on a costless cast and movemaskps */
inline uint32_t _mm256_movemask_epi32(__m256i v) {
    return _mm256_movemask_ps(_mm256_castsi256_ps(v));
}

// Generate Move mask via: _mm256_movemask_ps(_mm256_castsi256_ps(mask)); etc
// author: Froglegs, see https://stackoverflow.com/a/36949578
inline __m256i pack_left_epi32(__m256i values, uint32_t moveMask) {
    uint8_t *adr = g_pack_left_table_uint8_tx3 + moveMask * 3;
    __m256i indices = _mm256_set1_epi32(*reinterpret_cast<uint32_t*>(adr)); //lower 24 bits has our LUT

    // There is leftover data in the lane, but _mm256_permutevar8x32_ps  only examines the first 3 bits so this is ok
    __m256i shufmask = _mm256_srlv_epi32 (indices, _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21));
    return _mm256_permutevar8x32_epi32(values, shufmask);
}

inline __m256i pack_left_epi32(__m256i values, __m256i mask) {
    return pack_left_epi32(values, _mm256_movemask_epi32(mask));
}

inline __m256i _mm256_cmpgt_epu32(__m256i left, __m256i right) {
    __m256i  left_shifted = _mm256_xor_si256( left, _mm256_set1_epi32(0x80000000));
    __m256i right_shifted = _mm256_xor_si256(right, _mm256_set1_epi32(0x80000000));
    return _mm256_cmpgt_epi32(left_shifted, right_shifted);
}

/*
 * load a vector given an address which must be valid for a load of the vector size
 */
template <typename V>
inline V load(const void *);

template <>
inline __m256i load<__m256i>(const void *p) {
    return _mm256_loadu_si256(static_cast<const __m256i *>(p));
}

template <>
inline __m512i load<__m512i>(const void *p) {
    return _mm512_loadu_si512(static_cast<const __m512i *>(p));
}

template <typename V>
inline void store(void *p, V v);

template <>
inline void store(void *p, __m256i v) {
    _mm256_storeu_si256(static_cast<__m256i *>(p), v);
}

template <>
inline void store(void *p, __m512i v) {
    _mm512_storeu_si512(p, v);
}

template <typename V, typename T>
inline V to_simd(const std::vector<T>& in) {
    assert(in.size() * sizeof(T) == sizeof(V));
    return load<V>(in.data());
}

template <typename T = uint32_t, typename V>
inline std::vector<T> to_vector(V in) {
    static_assert(sizeof(V) % sizeof(T) == 0, "T size must divide V");
    std::vector<T> out(sizeof(V) / sizeof(T));
    store(out.data(), in);
    return out;
}

// template <typename T = std::uint32_t>
// inline std::vector<T> to_vector(__m256i in) { return to_vector_impl<T>(in); }

// template <typename T = std::uint32_t>
// inline std::vector<T> to_vector(__m512i in) { return to_vector_impl<T>(in); }

std::ostream& operator<<(std::ostream& os, __m256i v);

inline __m256i f2i(__m256 in) {
    return _mm256_castps_si256(in);
}

inline __m256 i2f(__m256i in) {
    return _mm256_castsi256_ps(in);
}

template <typename T, typename F>
std::vector<T> cvec(const std::vector<F>& in) {
    std::vector<T> ret;
    for (auto& e : in) {
        ret.push_back(static_cast<T>(e));
    }
    return ret;
}
/**
 * Full shifts - byte granular shifts across an entire 256-bit
 * vector.
 *
 * https://stackoverflow.com/a/25264853
 */
template <int N>
inline __m256i ss_mm256_sllx_si256(__m256i v) {
    static_assert(N >= 0 && N < 32, "N out of range");  // we could support a larger range
    if (N < 16) {
        auto t = _mm256_permute2x128_si256(v, v, _MM_SHUFFLE(0, 0, 2, 0));
        return _mm256_alignr_epi8(v, t, 16 - N);
    } else if (N == 16) {
        auto t = _mm256_permute2x128_si256(v, v, 0x8);
        // dbg(cvec<int>(to_vector<int8_t>(t)));
        return t;
    } else {
        return _mm256_slli_si256(_mm256_permute2x128_si256(v, v, 0x8), N - 16);
    }
}

#endif
