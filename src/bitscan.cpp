#include "accum7.hpp"
#include "bitscan.hpp"
#include "hedley.h"
#include "simd-support.hpp"

#include <immintrin.h>

namespace fastscancount {

using query_type = std::vector<uint32_t>;
using out_type = std::vector<uint32_t>;

#define UNROLL_X(fn, arg)\
        fn(0, arg);  \
        fn(1, arg);  \
        fn(2, arg);  \
        fn(3, arg);  \
        fn(4, arg);  \
        fn(5, arg);  \
        fn(6, arg);  \
        fn(7, arg);  \


template <typename E, typename D>
struct base_traits {
    using elem_type = E;
    using aux_type = bitscan_all_aux<E>;
    using btype = compressed_bitmap<E>;

    static constexpr size_t chunk_bits = btype::chunk_bits;

    template <size_t B>
    static constexpr bool has_override_middle = false;
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
void generic_populate_hits(std::vector<A>& accums, out_type& out, size_t offset) {
    for (auto& accum : accums) {
        auto satflags = accum.get_saturated();
        traits::populate_hits(satflags, offset, out);
        offset += traits::chunk_bits;
    }
}

template <size_t N, typename traits, typename A>
void handle_tail(
    size_t qidx,
    A& accums,
    const typename traits::aux_type& aux_info,
    const query_type& query,
    std::vector<const typename traits::btype *>& all_bitmaps,
    std::vector<const typename traits::elem_type*>& all_eptrs,
    size_t start_chunk,
    size_t end_chunk
    )
{
    assert(qidx + N == query.size());

    std::array<const typename traits::btype*, N> bitmaps;
    std::array<const typename traits::elem_type*, N> eptrs;

    std::copy(all_bitmaps.begin() + qidx, all_bitmaps.begin() + qidx + N, bitmaps.begin());
    std::copy(all_eptrs.begin() + qidx, all_eptrs.begin() + qidx + N, eptrs.begin());

    for (size_t c = start_chunk; c < end_chunk; c++) {
        for (size_t i = 0; i < N; i++) {
            auto e = traits::expand(*bitmaps[i], c, eptrs[i]);
            assert(c - start_chunk < accums.size());
            accums[c - start_chunk].accept(e);
        }
    }

    // copy the eptrs back for the next pass
    std::copy(eptrs.begin(), eptrs.end(), all_eptrs.begin() + qidx);
}

template <typename traits, size_t B>
HEDLEY_NEVER_INLINE
void handle_middle(
        std::vector<typename traits::template accum_type<B>>& accums,
        typename traits::btype const * const * bitmaps,
        typename traits::elem_type const * * eptrs,
        size_t start_chunk,
        size_t end_chunk
    )
{
    if constexpr (traits::template has_override_middle<B>) {
        DBG(printf("Using override_middle for %zu %s\n", B, __PRETTY_FUNCTION__));
        traits::override_middle(accums, bitmaps, eptrs, start_chunk, end_chunk);
        return;
    }
    #define DEF_EPTR(i,_) typename traits::elem_type const *eptr_##i = eptrs[i];
    UNROLL_X(DEF_EPTR,_);

    for (size_t c = start_chunk; c < end_chunk; c++) {
        #define BODY(i,_) auto e##i = traits::expand(*bitmaps[i], c, eptr_##i);
        UNROLL_X(BODY,_);

        assert(c - start_chunk < accums.size());
        accums[c - start_chunk].accept8(e0, e1, e2, e3, e4, e5, e6, e7);
    }

    // copy the eptrs back for the next pass
    #define ASSIGN_EPTR(i,_) eptrs[i] = eptr_##i;
    UNROLL_X(ASSIGN_EPTR,_);
}

#define MIDDLE_ARGS(b)                                             \
        std::vector<accumulator<b, __m512i, m512_traits>>& accums, \
        compressed_bitmap<uint32_t> const * const * bitmaps,       \
        uint32_t const ** eptrs,                                   \
        size_t start_chunk,                                        \
        size_t end_chunk                                           \

#ifdef __AVX512F__

struct avx512_traits_asm : avx512_traits<uint32_t> {
    template <size_t B>
    static constexpr bool has_override_middle = false;

    template <size_t B>
    HEDLEY_NEVER_INLINE
    static void override_middle( MIDDLE_ARGS(B) );
};

template<>
constexpr bool avx512_traits_asm::has_override_middle<3> = true;

/**
 * Add -fno-stack-protector
 */
extern "C"
void override_middle_asm_3( MIDDLE_ARGS(3) );
// {
//     handle_middle<avx512_traits<uint32_t>, 3>(accums, bitmaps, eptrs, start_chunk, end_chunk);
// }

template <>
void avx512_traits_asm::override_middle<3>( MIDDLE_ARGS(3) ) {
    override_middle_asm_3(accums, bitmaps, eptrs, start_chunk, end_chunk);
}



#endif

template <size_t THRESHOLD, typename traits>
void bitscan_generic(out_type& out,
                     const typename traits::aux_type& aux_info,
                     const std::vector<uint32_t>& query)
{
    constexpr size_t B = lg2_up(THRESHOLD + 1); // number of bits needed in the accumulators
    using T = typename traits::elem_type;
    using atype = typename traits::template accum_type<B>;

    assert(atype::max >= THRESHOLD + 1u); // need to increase A_BITS if this fails

    const size_t array_count = query.size();
    const size_t total_chunk_count = aux_info.get_chunk_count();
    constexpr size_t chunks_per_pass = 128;

    atype accum_init(atype::max - THRESHOLD - 1);

    std::vector<const compressed_bitmap<T>*> all_bitmaps;
    std::vector<const T*> all_eptrs;
    all_bitmaps.resize(array_count);
    all_eptrs.resize(array_count);

    for (size_t qidx = 0; qidx < array_count; qidx++) {
        auto did = query.at(qidx);
        assert(did < aux_info.bitmaps.size());
        all_bitmaps[qidx] = &aux_info.bitmaps[did];;
        all_eptrs[qidx]   = all_bitmaps[qidx]->elements.data();
        assert(all_bitmaps[qidx]->chunk_count() == total_chunk_count);
        assert(all_bitmaps[qidx] && all_eptrs[qidx]);
    }

    std::vector<atype> accums;
    // accums.resize(pass_chunk_count, accum_init);

    for (size_t start_chunk = 0; start_chunk < total_chunk_count; start_chunk += chunks_per_pass) {

        const size_t pass_chunk_count = std::min(chunks_per_pass, total_chunk_count - start_chunk);
        const size_t end_chunk = start_chunk + pass_chunk_count;

        accums.assign(pass_chunk_count, accum_init);

        // we take queries in blocks of 8
        constexpr size_t stream_count = 8;
        size_t qidx = 0;
        for (; qidx + stream_count <= array_count; qidx += stream_count) {
            auto bitmaps = &all_bitmaps.at(qidx);
            auto eptrs = &all_eptrs.at(qidx);
            handle_middle<traits, B>(accums, bitmaps, eptrs, start_chunk, end_chunk);
        }

        // TODO get rid of this ugly switch
        size_t rem = array_count - qidx;
        switch (rem) {
            case 0: break;
            case 1: handle_tail<1, traits>(qidx, accums, aux_info, query, all_bitmaps, all_eptrs, start_chunk, end_chunk); break;
            case 2: handle_tail<2, traits>(qidx, accums, aux_info, query, all_bitmaps, all_eptrs, start_chunk, end_chunk); break;
            case 3: handle_tail<3, traits>(qidx, accums, aux_info, query, all_bitmaps, all_eptrs, start_chunk, end_chunk); break;
            case 4: handle_tail<4, traits>(qidx, accums, aux_info, query, all_bitmaps, all_eptrs, start_chunk, end_chunk); break;
            case 5: handle_tail<5, traits>(qidx, accums, aux_info, query, all_bitmaps, all_eptrs, start_chunk, end_chunk); break;
            case 6: handle_tail<6, traits>(qidx, accums, aux_info, query, all_bitmaps, all_eptrs, start_chunk, end_chunk); break;
            default: assert(false);
        }


        generic_populate_hits<traits>(accums, out, start_chunk * traits::chunk_bits);
    }
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

#ifdef __AVX512F__
void bitscan_avx512_asm(const data_ptrs &, std::vector<uint32_t> &out,
                    uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                    const std::vector<uint32_t>& query)
{
    if (threshold >= MAX_T) throw std::runtime_error("MAX_T too small");
    lut_holder<avx512_traits_asm>::lut[threshold](out, aux_info, query);
}
#endif

/* explicit instantiations */

template void bitscan_fake2<uint32_t>(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                const std::vector<uint32_t>& query);

template void bitscan_avx512<uint32_t>(const data_ptrs &, std::vector<uint32_t> &out,
                uint8_t threshold, const bitscan_all_aux<uint32_t>& aux_info,
                const std::vector<uint32_t>& query);

}
