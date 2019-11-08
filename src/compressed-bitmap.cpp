
#include "compressed-bitmap.hpp"
#include "common.h"
#include "boost/dynamic_bitset.hpp"

#include <algorithm>
#include <assert.h>

using dynbits = boost::dynamic_bitset<>;

std::string to_string(const dynbits &in) {
    std::string str;
    to_string(in, str);
    std::reverse(str.begin(), str.end());
    return str;
}

template<std::size_t N>
std::string to_string(const std::bitset<N> &in) {
    std::string str = in.to_string();
    std::reverse(str.begin(), str.end());
    return str;
}

dynbits subset(const dynbits& in, size_t from, size_t to) {
    assert(to <= in.size());
    assert(from <= to);
    auto ret = in >> from;
    ret.resize(to - from);
    return ret;
}

template <typename T>
inline size_t tzcnt(T val) {
    assert(val);
    static_assert(sizeof(T) <= sizeof(unsigned long));
    return __builtin_ctzl(val);
}

template <typename T>
typename compressed_bitmap<T>::chunk_type compressed_bitmap<T>::expand(size_t idx, const T*& eptr) const {
    assert(idx < chunk_count());
    assert(eptr >= elements.data());
    assert(eptr <= elements.data() + elements.size());
    chunk_type chunk;
    auto c = control.at(idx);
    while (c) {
        assert(eptr < elements.data() + elements.size());
        auto subchunk_idx = tzcnt(c & -c) * subchunk_bits;
        assert(subchunk_idx < chunk_bits);
        auto elem = *eptr++;
        assert(elem); // zero element doens't make sense
        while (elem) {
            auto elem_idx = tzcnt(elem);
            assert(subchunk_idx + elem_idx < chunk_bits);
            chunk.set(subchunk_idx + elem_idx);
            elem &= elem - 1; // clear lowest set bit
        }
        c &= c - 1; // clear lowest set bit
    }
    return chunk;
}


/**
 * T the type of the output data entries, either uint32_t or uint8_t I guess
 */
template <typename T>
compressed_bitmap<T>::compressed_bitmap(const std::vector<uint32_t>& array, uint32_t largest) {
    assert(!array.empty()); // some bugs for empty arrays
    constexpr auto bits_per_entry = sizeof(T) * 8;
    static_assert(chunk_bits % bits_per_entry == 0);
    constexpr auto control_bits = 8 * sizeof(control[0]);

    if (largest == -1u) {
        largest = array.back();
    }

    for (size_t i = 0, lower_bound = 0; lower_bound <= largest; lower_bound += chunk_bits) {
        size_t upper_bound = lower_bound + chunk_bits;
        chunk_type chunk;
        DBG(printf("Chunk %zu - %zu, elems: ", lower_bound, upper_bound));
        size_t elem_count = 0;
        while (i < array.size() && array.at(i) < upper_bound) {
            auto e = array[i];
            assert(e >= lower_bound);
            assert(e < upper_bound);
            chunk.set(e - lower_bound);
            i++;
            elem_count++;
            DBG(printf("%zu ", e - lower_bound));
        }
        DBG(printf("\nbitmap R: %s\n", to_string(chunk).c_str()));
        assert(chunk.count() == elem_count);

        std::bitset<control_bits> one_control;
#ifndef NDEBUG
        size_t size_before = elements.size();
#endif
        // for each bits_per_entry chunk in the bitmap
        for (size_t suboffset = 0, bit = 0; suboffset < chunk_bits; suboffset += bits_per_entry, bit++) {
            auto sub = subset(chunk, suboffset, suboffset + bits_per_entry);
            DBG(printf("subset: %s (%szero)\n", to_string(sub).c_str(), sub.any() ? "non" : ""));
            if (sub.any()) {  // any bit set in this subchunk?
                elements.push_back(sub.to_ulong());
                one_control.set(bit);
            }
        }

        // number of bits set in the control must be the number of elements addded
        DBG(printf("elem_count: %zu required: %zu one_control: %s\n", elem_count,
                elements.size() - size_before, to_string(one_control).c_str()));
        assert(elem_count >= elements.size() - size_before);
        assert(__builtin_popcountl(one_control.to_ulong()) == elements.size() - size_before);
        control.push_back(one_control.to_ulong());
    }

    // printf("carray size: %zu expected %zu\n", control.size(), div_up((size_t)(array.back() + 1), chunk_bits));
    assert(elements.size() <= array.size());
    assert(control.size() == div_up((size_t)(largest + 1), chunk_bits) );
    DBG(printf("arr size: %zu control size: %zu elem size %zu\n", array.size(), control.size(), elements.size()));

    // add some buffer to the end of elements, so overreads don't fail
    // this isn't the right way to do it, but works in practice and will
    // pass valgrind (and ASAN I think)
    elements.reserve(elements.size() + 64 / sizeof(T));
}

template <typename T>
std::vector<size_t> compressed_bitmap<T>::indices() const {
    std::vector<size_t> ret;
    size_t base = 0, didx = 0;
    for (auto& chunk : chunks()) {
        auto pos = chunk.find_first();
        while (pos != chunk_type::npos) {
            // assert(pos < chunk.size());
            ret.push_back(base + pos);
            pos = chunk.find_next(pos);
        }
        base += chunk_bits;
    }
    return ret;
}

template <typename T>
std::vector<typename compressed_bitmap<T>::chunk_type> compressed_bitmap<T>::chunks() const {
    std::vector<chunk_type> ret;
    size_t didx = 0;
    for (auto c : control) {
        chunk_type chunk;
        while (c) {
            auto subchunk_idx = tzcnt(c & -c) * subchunk_bits;
            assert(subchunk_idx < chunk_bits);
            auto elem = elements.at(didx++);
            assert(elem); // zero element doens't make sense
            while (elem) {
                auto elem_idx = tzcnt(elem);
                assert(subchunk_idx + elem_idx < chunk_bits);
                chunk.set(subchunk_idx + elem_idx);
                elem &= elem - 1; // clear lowest set bit
            }
            c &= c - 1; // clear lowest set bit
        }
        ret.push_back(chunk);
    }
    return ret;
}

template class compressed_bitmap<uint32_t>;
