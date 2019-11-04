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

    using btype = compressed_bitmap<T>;

    std::vector<accumulator<7, __m512i, m512_traits>> accums;
    accumulator<7, __m512i, m512_traits> accumz;
    accums.resize(aux_info.get_chunk_count());

    for (auto did : query) {
        auto& bitmap = aux_info.bitmaps.at(did);

        const T *eptr = bitmap.elements.data();
        for (size_t c = 0, sz = bitmap.control.size(); c < sz; c++) {
            auto control = bitmap.control.at(c);
            auto mask = _load_mask16(&control);
            auto data = _mm512_load_si512(eptr);
            auto expanded = _mm512_maskz_expand_epi32(mask, data);
            accums.at(c).accept(expanded);
            eptr += __builtin_popcount(mask);
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

#endif
}

template void bitscan_avx512<uint32_t>(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                const std::vector<uint32_t>& query);

}
