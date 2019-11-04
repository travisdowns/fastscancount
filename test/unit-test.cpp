/*
 * unit-test.cpp
 */

#include "catch.hpp"

#include "compressed-bitmap.hpp"

#include "bitscan.hpp"

using vst = std::vector<size_t>;
using cb32 = compressed_bitmap<uint32_t>;

TEST_CASE( "compressed-bitmap-size" ) {
    compressed_bitmap<uint32_t> cb({100});

    CHECK(cb.byte_size() == 6);

    cb = {{1000}};

    CHECK(cb.byte_size() == 8);  // 1000 > 512, so we need an empty control
}

TEST_CASE( "compressed-bitmap-indices" ) {

    // 1 elem
    REQUIRE(cb32({1}).indices()  == vst{1});
    REQUIRE(cb32({35}).indices() == vst{35});
    REQUIRE(cb32({550}).indices() == vst{550});

    // 2 elem
    REQUIRE(cb32({1, 2}).indices()   == vst{1, 2});
    REQUIRE(cb32({1, 35}).indices()  == vst{1, 35});
    REQUIRE(cb32({1, 550}).indices() == vst{1, 550});
}

TEST_CASE( "compressed-bitmap-chunk" ) {

    cb32 cb({999});

    cb = cb32({1, 3});
    auto chunks = cb.chunks();

    REQUIRE(chunks.size() == 1);
    auto chunk = chunks.at(0);
    REQUIRE(chunk.count() == 2);
    REQUIRE(chunk.test(1));
    REQUIRE(chunk.test(3));

    cb = cb32({513, 515});
    chunks = cb.chunks();

    REQUIRE(chunks.size() == 2);
    REQUIRE(chunks.at(0).count() == 0);
    chunk = chunks.at(1);
    REQUIRE(chunk.count() == 2);
    REQUIRE(chunk.test(1));
    REQUIRE(chunk.test(3));
}

using namespace fastscancount;

template <size_t used_bits = 1>
struct int_traits : default_traits<uint32_t> {

    static void check_valid(uint32_t v) {
        assert((v >> used_bits) == 0);
    }

    static uint32_t not_(uint32_t v) {
        return ~v & ((1u << used_bits) - 1);
    }

    static bool test(uint32_t v, size_t idx) {
        assert(idx < used_bits);
        check_valid(v);
        return v & ((uint32_t)1 << idx);
    }

    static size_t size() {
        return used_bits;
    }

    static bool zero(uint32_t v) {
        check_valid(v);
        return v == 0;
    }
};

template <typename T>
void accum_test()
{
    {
        auto accum = T::template make<2, 1>();

        REQUIRE(accum.get_sums() == vst{0});
        accum.accept(0);
        REQUIRE(accum.get_sums() == vst{0});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{1});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{2});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{3});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{4});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{4}); // saturated

    }

    {
        auto accum = T::template make<2, 2>();

        REQUIRE(accum.get_sums() == vst{0, 0});
        accum.accept(2);
        REQUIRE(accum.get_sums() == vst{0, 1});
        accum.accept(3);
        REQUIRE(accum.get_sums() == vst{1, 2});
        accum.accept(3);
        REQUIRE(accum.get_sums() == vst{2, 3});
        accum.accept(3);
        REQUIRE(accum.get_sums() == vst{3, 4});
        accum.accept(3);
        REQUIRE(accum.get_sums() == vst{4, 4});
    }

    {
        auto accum = T::template make<2, 2>();

        accum = T::template make<2, 2>(1);
        REQUIRE(accum.get_sums() == vst{1, 1});

        accum = T::template make<2, 2>(2);
        REQUIRE(accum.get_sums() == vst{2, 2});

        accum = T::template make<2, 2>(3);
        REQUIRE(accum.get_sums() == vst{3, 3});

        accum = T::template make<2, 2>(10);
        REQUIRE(accum.get_sums() == vst{4, 4});
    }

    {
        auto accum = T::template make<2, 2>();

        REQUIRE(accum.get_saturated() == 0b00);

        accum.accept(0b10);
        REQUIRE(accum.get_saturated() == 0b00);
        accum.accept(0b11); // 2 1
        REQUIRE(accum.get_saturated() == 0b00);
        accum.accept(0b11); // 3 2
        REQUIRE(accum.get_saturated() == 0b00);
        accum.accept(0b11); // 4 3
        REQUIRE(accum.get_saturated() == 0b10);
        accum.accept(0b11); // 4 4
        REQUIRE(accum.get_saturated() == 0b11);
        accum.accept(0b11); // 4 4
        REQUIRE(accum.get_saturated() == 0b11);
        accum.accept(0b00); // 4 4
        REQUIRE(accum.get_saturated() == 0b11);
    }


}

struct int_holder {
    template <size_t B, size_t C>
    static auto make(size_t initial = 0) {
        return accumulator<B, int, int_traits<C>>(initial);
    }
};

TEST_CASE("accumulator") {
    accum_test<int_holder>();
}

/**
 * wrapper for accumulator<B, __m512i, m512_traits> so that
 * accept takes a plain integer like the version above, so
 * the tests can be shared.
 *
 * The counter is limited to C counters for sanity.
 */
template <size_t B, size_t C = 1>
struct accum512 : fastscancount::accumulator<B, __m512i, fastscancount::m512_traits> {
    static_assert(C <= 32, "C too big"); // otherwise stuff that returns uint32_t won't work
    using base = fastscancount::accumulator<B, __m512i, fastscancount::m512_traits>;

    accum512(size_t initial) : base{initial} {}

    void accept(int v) {
        auto v512 = _mm512_set_epi32(
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, v
        );
        base::accept(v512);
    }

    std::vector<size_t> get_sums() {
        std::vector<size_t> ret, b = base::get_sums();
        assert(C <= b.size());
        std::copy(b.begin(), b.begin() + C, std::back_inserter(ret));
        return ret;
    }

    uint32_t get_saturated() {
        auto vec = base::get_saturated();
        return _mm256_extract_epi32(_mm512_castsi512_si256(vec), 0);
    }
};

struct accum512_holder {
    template <size_t B, size_t C>
    static auto make(size_t initial = 0) {
        return accum512<B, C>(initial);
    }
};


#ifdef __AVX512F__
TEST_CASE("accumulator-512") {
    accum_test<accum512_holder>();
}
#endif
