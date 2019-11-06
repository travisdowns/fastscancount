#include "accum7.hpp"
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
    using btype = compressed_bitmap<T>;
    using atype = accum7<__m512i, m512_traits>;

    assert(atype::max >= threshold + 1u); // need to increase A_BITS if this fails

    const size_t chunk_count = aux_info.get_chunk_count();

    atype accum_init(atype::max - threshold - 1);
    std::vector<atype> accums;
    accums.resize(chunk_count, accum_init);

    // first we take queries in chunks of 7
    constexpr size_t stream_count = 7;
    size_t qidx = 0;
    for (; qidx + stream_count <= query.size();) {
        assert(qidx < query.size());
        auto didx = query.at(qidx);

        std::array<const compressed_bitmap<T>*, stream_count> bitmaps;
        std::array<const T*, stream_count> eptrs;
        bitmaps.fill(nullptr);

        for (size_t i = 0; i < stream_count; i++) {
            auto did = query.at(qidx++);
            auto& bitmap = aux_info.bitmaps.at(did);
            bitmaps[i] = &bitmap;
            eptrs[i] = bitmap.elements.data();
            assert(bitmaps[i]->chunk_count() == chunk_count);
        }

        for (size_t c = 0; c < chunk_count; c++) {
            #define UNROLL(i) auto e##i = bitmaps[i]->expand512(c, eptrs[i]);

            UNROLL(0);
            UNROLL(1);
            UNROLL(2);
            UNROLL(3);
            UNROLL(4);
            UNROLL(5);
            UNROLL(6);

            #undef UNROLL

            accums.at(c).accept7(e0, e1, e2, e3, e4, e5, e6);
        }
    }

    // remaining queries one-by-one
    for (; qidx < query.size(); qidx++) {
        auto& bitmap = aux_info.bitmaps.at(query.at(qidx));
        const T *eptr = bitmap.elements.data();
        for (size_t c = 0; c < chunk_count; c++) {
            auto expanded = bitmap.expand512(c, eptr);
            accums.at(c).accept(expanded);
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

// template <typename T = uint32_t>
// inline std::array<T, sizeof(V) / sizeof(T)> to_vector(const boost::dynamic_bitset& in) {
//     assert(in.size() % (sizeof(T) * 8) == 0, "T size must divide bitset.size()");
//     std::vector<T> ret(in.size() / (sizeof(T) * 8));
//     for (size_t e = 0; e < ret.size(); e++) {
//         T elem = 0;
//         for (size_t i = 0; i )
//     }
//     return out;
// }


template <typename T>
void bitscan_fake2(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<T>& aux_info,
                const std::vector<uint32_t>& query)
{
    using btype = compressed_bitmap<T>;
    using ctype = typename btype::chunk_type;
    using atype = accum7<ctype, chunk_traits<T>>;

    // DBG(printf("atype::max %zu thresh: %u\n", atype::max, threshold));
    assert(atype::max >= threshold + 1u); // need to increase A_BITS if this fails

    const size_t chunk_count = aux_info.get_chunk_count();

    atype accum_init(atype::max - threshold - 1);
    // printf("init : ");
    // for (auto s : accum_init.get_sums()) {
    //     printf("%zu ", s);
    // }
    // printf("\n");
    std::vector<atype> accums;
    accums.resize(chunk_count, accum_init);

    // first we take queries in chunks of 7
    constexpr size_t stream_count = 7;
    size_t qidx = 0;
    for (; qidx + stream_count <= query.size();) {
        assert(qidx < query.size());
        auto didx = query.at(qidx);

        std::array<const compressed_bitmap<T>*, stream_count> bitmaps;
        std::array<const T*, stream_count> eptrs;
        bitmaps.fill(nullptr);

        for (size_t i = 0; i < stream_count; i++) {
            auto did = query.at(qidx++);
            auto& bitmap = aux_info.bitmaps.at(did);
            bitmaps[i] = &bitmap;
            eptrs[i] = bitmap.elements.data();
            assert(bitmaps[i]->chunk_count() == chunk_count);
        }

        for (size_t c = 0; c < chunk_count; c++) {
            #define UNROLL(i) auto e##i = bitmaps[i]->expand(c, eptrs[i]);

            UNROLL(0);
            UNROLL(1);
            UNROLL(2);
            UNROLL(3);
            UNROLL(4);
            UNROLL(5);
            UNROLL(6);

            #undef UNROLL

            accums.at(c).accept7(e0, e1, e2, e3, e4, e5, e6);
        }
    }

    // remaining queries one-by-one
    for (; qidx < query.size(); qidx++) {
        auto& bitmap = aux_info.bitmaps.at(query.at(qidx));
        const T *eptr = bitmap.elements.data();
        for (size_t c = 0; c < chunk_count; c++) {
            auto expanded = bitmap.expand(c, eptr);
            accums.at(c).accept(expanded);
        }
    }

    size_t offset = 0;
    for (auto& accum : accums) {
        // for (auto s : accum.get_sums()) {
        //     printf("%zu ", s);
        // }
        // printf("\n");
        auto flags = accum.get_saturated();
        // printf("sat had %zu bits\n", flags.count());
        for (size_t i = 0; i < btype::chunk_bits; i++) {
            if (flags.test(i)) {
                out.push_back(i + offset);
            }
        }
        offset += btype::chunk_bits;
    }
}

/* explicit instantiations */

template void bitscan_fake2<uint32_t>(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                const std::vector<uint32_t>& query);

template void bitscan_avx512<uint32_t>(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                const std::vector<uint32_t>& query);

}
