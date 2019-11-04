#include "bitscan.hpp"

#include <immintrin.h>

namespace fastscancount {


template <typename T>
void bitscan_avx512(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<T>& aux_info,
                const std::vector<uint32_t>& query)
{
#ifndef __AVX512F__
    throw std::runtime_error("not compiled for AVX-512");
#else
    std::vector<uint8_t> counters(aux_info.largest + 1);
    using btype = compressed_bitmap<T>;

    for (auto did : query) {
        auto bitmap = aux_info.bitmaps.at(did);
        size_t chunk_offset = 0;

        const T *eptr = bitmap.elements.data();
        for (auto& control : bitmap.control) {
            auto mask = _load_mask16(&control);
            auto data = _mm512_load_si512(eptr);
            auto expanded = _mm512_maskz_expand_epi32(mask, data);
            auto chunk = to_bitset(expanded);
            auto pos = chunk.find_first();
            while (pos != btype::chunk_type::npos) {
                ++counters.at(chunk_offset + pos);
                pos = chunk.find_next(pos);
            }
            chunk_offset += btype::chunk_bits;
            eptr += __builtin_popcount(mask);
        }
    }

    for (size_t i = 0; i < counters.size(); i++) {
        if (counters.at(i) > threshold) {
            out.push_back(i);
        }
    }
#endif
}

template void bitscan_avx512<uint32_t>(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                const std::vector<uint32_t>& query);

}
