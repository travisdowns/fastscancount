#include "accum7.hpp"
#include "bitscan.hpp"
#include "hedley.h"
#include "simd-support.hpp"

#include <immintrin.h>

namespace fastscancount {

using query_type = std::vector<uint32_t>;
using out_type = std::vector<uint32_t>;


template <typename E, typename D>
struct base_traits {
    using elem_type = E;
    using aux_type = bitscan_all_aux<E>;
    using btype = compressed_bitmap<E>;

    static constexpr size_t chunk_bits = btype::chunk_bits;
};

/*
 * The element type of the underlying bitmap.
 */
template <typename E>
struct fake_traits : base_traits<E, fake_traits<E>> {
    using base = base_traits<E, fake_traits<E>>;
    using chunk_type = typename base::btype::chunk_type;

    template <size_t B>
    using accum_type = accumulator<B, chunk_type, chunk_traits<E>>;

    static chunk_type expand(const typename base::btype& bitmap, size_t index, const E*& eptr) {
        return bitmap.expand(index, eptr);
    }

    static void populate_hits(const chunk_type& flags, uint32_t offset, out_type& out) {
            // printf("sat had %zu bits\n", flags.count());
        for (size_t i = 0; i < base::chunk_bits; i++) {
            if (flags.test(i)) {
                out.push_back(offset + i);
            }
        }
    }
};

#ifdef __AVX512F__

template <typename E>
struct avx512_traits : base_traits<E, avx512_traits<E>> {
    using base = base_traits<E, avx512_traits<E>>;
    using chunk_type = __m512i;

    template <size_t B>
    using accum_type = accumulator<B, chunk_type, m512_traits>;

    static chunk_type expand(const typename base::btype& bitmap, size_t index, const E*& eptr) {
        return bitmap.expand512(index, eptr);
    }

    static void populate_hits(const chunk_type& flags, uint32_t offset, out_type& out) {
        auto flags64 = to_array<uint64_t>(flags);
        assert(flags64.size() * 64 == base::chunk_bits);
        for (auto f : flags64) {
            while (f) {
                uint32_t idx = __builtin_ctzl(f);
                out.push_back(idx + offset);
                f &= (f - 1);
            }
            offset += 64;
        }
    }
};

#endif

template <typename traits, typename A>
HEDLEY_NEVER_INLINE
void generic_populate_hits(std::vector<A>& accums,
                           out_type& out) {
    size_t offset = 0;
    for (auto& accum : accums) {
        auto satflags = accum.get_saturated();
        traits::populate_hits(satflags, offset, out);
        offset += traits::chunk_bits;
    }
}

template <size_t N, typename traits, typename A>
void handle_tail(size_t qidx, A& accums, const typename traits::aux_type& aux_info, const query_type& query) {
    size_t chunk_count = aux_info.get_chunk_count();
    for (; qidx < query.size(); qidx++) {
        auto& bitmap = aux_info.bitmaps.at(query.at(qidx));
        const typename traits::elem_type *eptr = bitmap.elements.data();
        for (size_t c = 0; c < chunk_count; c++) {
            auto expanded = traits::expand(bitmap, c, eptr);
            assert(c < accums.size());
            accums[c].accept(expanded);
        }
    }
}

template <size_t THRESHOLD, typename traits>
void bitscan_generic(out_type& out,
                     const typename traits::aux_type& aux_info,
                     const std::vector<uint32_t>& query)
{
    using T = typename traits::elem_type;
    using atype = typename traits::template accum_type<lg2_up(THRESHOLD + 1)>;

    assert(atype::max >= THRESHOLD + 1u); // need to increase A_BITS if this fails

    const size_t chunk_count = aux_info.get_chunk_count();

    atype accum_init(atype::max - THRESHOLD - 1);
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
            assert(did < aux_info.bitmaps.size());
            auto& bitmap = aux_info.bitmaps[did];
            bitmaps[i] = &bitmap;
            eptrs[i] = bitmap.elements.data();
            assert(bitmaps[i]->chunk_count() == chunk_count);
        }

        for (size_t c = 0; c < chunk_count; c++) {
            #define UNROLL(i) auto e##i = traits::expand(*bitmaps[i], c, eptrs[i]);

            UNROLL(0);
            UNROLL(1);
            UNROLL(2);
            UNROLL(3);
            UNROLL(4);
            UNROLL(5);
            UNROLL(6);

            #undef UNROLL

            assert(c < accums.size());
            accums[c].accept7(e0, e1, e2, e3, e4, e5, e6);
        }
    }

    handle_tail<0, traits>(qidx, accums, aux_info, query);

    generic_populate_hits<traits>(accums, out);
}


template <typename traits>
using bitscan_fn = void (out_type& out,
                         const typename traits::aux_type& aux_info,
                         const std::vector<uint32_t>& query);


template <typename traits, size_t I, size_t MAX>
constexpr void make_helper(std::array<bitscan_fn<traits> *, MAX>& a) {
    if constexpr (I < MAX) {
        a[I] = bitscan_generic<I, traits>;
        make_helper<traits, I + 1, MAX>(a);
    }
}

template <typename traits, size_t MAX>
constexpr std::array<bitscan_fn<traits> *, MAX> make_lut() {
    std::array<bitscan_fn<traits> *, MAX> ret{};
    make_helper<traits, 1, MAX>(ret);
    return ret;
}

static constexpr size_t MAX_T = 11;

template <typename traits>
struct lut_holder {
    static constexpr std::array<bitscan_fn<traits> *, MAX_T> lut = make_lut<traits, MAX_T>();
};

template <typename E>
void bitscan_avx512(const data_ptrs &, std::vector<uint32_t> &out,
                    uint8_t threshold, const bitscan_all_aux<E>& aux_info,
                    const std::vector<uint32_t>& query)
{
#ifndef __AVX512F__
    throw std::runtime_error("not compiled for AVX-512");
#else
    if (threshold >= MAX_T) throw std::runtime_error("MAX_T too small");
    lut_holder<avx512_traits<E>>::lut[threshold](out, aux_info, query);
#endif
}

template <typename E>
void bitscan_fake2(const data_ptrs &, std::vector<uint32_t> &out,
                   uint8_t threshold, const bitscan_all_aux<E>& aux_info,
                   const std::vector<uint32_t>& query)
{
    if (threshold >= MAX_T) throw std::runtime_error("MAX_T too small");
    lut_holder<fake_traits<E>>::lut[threshold](out, aux_info, query);
}

/* explicit instantiations */

template void bitscan_fake2<uint32_t>(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                const std::vector<uint32_t>& query);

template void bitscan_avx512<uint32_t>(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                const std::vector<uint32_t>& query);

}
