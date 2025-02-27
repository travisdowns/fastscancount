#ifndef COMPRESSED_BITMAP_H_
#define COMPRESSED_BITMAP_H_

#include "fastbitset.hpp"
#include "hedley.h"

#include <immintrin.h>
#include <inttypes.h>
#include <vector>
#include <bitset>



template <typename T> struct control_for {};

template<> struct control_for<uint8_t> { using type = uint64_t; };

template<> struct control_for<uint32_t> { using type = uint16_t; };

/**
 * Compressed bitmap using T as the element type.
 */
template <typename T>
struct compressed_bitmap {
    static constexpr size_t chunk_bits = 512;
    static constexpr size_t subchunk_bits = sizeof(T) * 8;

    using fixed_chunk = fastbitset<chunk_bits>;
    using chunk_type = fixed_chunk;

    using control_type = typename control_for<T>::type;

    std::vector<control_type> control;
    std::vector<T> elements;

    compressed_bitmap(const std::vector<uint32_t>& array, uint32_t largest = -1);

    /**
     * Decode the bitmap to a list of all set indices. This is not going to be the
     * way you want to use this if you care about performance, but it is useful
     * for writing naive algorithms.
     */
    std::vector<size_t> indices() const;

    /**
     * Return a vector of all chunks.
     */
    std::vector<chunk_type> chunks() const;

    /**
     * How many chunks are in this bitmap (not that this may include trailing all-zero chunks).
     */
    size_t chunk_count() const {
        return control.size();
    }

    chunk_type expand(size_t idx, const T*& eptr) const;

#ifdef __AVX512F__
    /**
     * Expand one chunk given its index and an element pointer (which will be udpated by this call).
     */
    HEDLEY_ALWAYS_INLINE
    __m512i expand512(size_t idx, const T*& eptr) const {
        assert(idx < chunk_count());
        assert(eptr >= elements.data());
        assert(eptr + 64/sizeof(*eptr) <= elements.data() + elements.capacity());  // naughty check that our read is valid
        // if (!(eptr + 64/sizeof(*eptr) <= elements.data() + elements.capacity())) {
        //     printf("eptr %p s: %p e: %p ec: %p\n", eptr, elements.data(),
        //             elements.data() + elements.size(), elements.data() + elements.capacity());
        //     assert(false);
        // }
        auto mask = _load_mask16(const_cast<control_type *>(control.data()) + idx);
        auto data = _mm512_loadu_si512(eptr);
        auto expanded = _mm512_maskz_expand_epi32(mask, data);
        eptr += __builtin_popcountl(mask);
        return expanded;
    }
#endif

    /**
     * The size in byte of the bitmap (not counting e.g., std::vector overhead, etc).
     */
    size_t byte_size() const {
        return control.size() * sizeof(control[0]) + elements.size() * sizeof(elements[0]);
    }
};

#endif
