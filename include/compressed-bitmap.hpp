#ifndef COMPRESSED_BITMAP_H_
#define COMPRESSED_BITMAP_H_

#include "boost/dynamic_bitset.hpp"

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

    struct chunk_type : boost::dynamic_bitset<> {
        chunk_type() : boost::dynamic_bitset<>(chunk_bits) {}
        static size_t size() { return chunk_bits; }
        const boost::dynamic_bitset<>& as_bitset() const { return static_cast<const boost::dynamic_bitset<>&>(*this); }
    };

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

    /**
     * The size in byte of the bitmap (not counting e.g., std::vector overhead, etc).
     */
    size_t byte_size() const {
        return control.size() * sizeof(control[0]) + elements.size() * sizeof(elements[0]);
    }
};

#endif
