#include "bitscan.hpp"
#include "simd-support.hpp"

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
    constexpr size_t A_BITS = 4;
    using atype = accumulator<A_BITS, __m512i, m512_traits>;
    using btype = compressed_bitmap<T>;

    DBG(printf("atype::max %zu thresh: %u\n", atype::max, threshold));
    assert(atype::max >= threshold + 1u); // need to increase A_BITS if this fails
    atype accum_init(atype::max - threshold - 1);
    std::vector<atype> accums;
    accums.resize(aux_info.get_chunk_count(), accum_init);

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
        __m512i flags = accum.get_saturated();
        auto flags64 = to_array<uint64_t>(flags);
        assert(flags64.size() * 64 == btype::chunk_bits);
        for (auto f : flags64) {
            while (f) {
                uint32_t idx = __builtin_ctzl(f);
                out.push_back(idx + offset);
                f &= (f - 1);
            }
            offset += 64;
        }
    }

#endif
}

template void bitscan_avx512<uint32_t>(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                const std::vector<uint32_t>& query);

}
