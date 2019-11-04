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

template <size_t used_bits = 1>
struct int_traits : fastscancount::default_traits<uint32_t> {

    static void check_valid(uint32_t v) {
        assert((v >> used_bits) == 0);
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

TEST_CASE("accumulator") {
    using namespace fastscancount;
    {
        accumulator<2, int, int_traits<>> accum;

        REQUIRE(accum.get_sums() == vst{0});
        accum.accept(0);
        REQUIRE(accum.get_sums() == vst{0});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{1});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{2});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{3});

        try {
            accum.accept(1);
            FAIL("should have thrown on overflow");
        } catch (std::runtime_error &) {
        }
    }

    {
        accumulator<2, int, int_traits<2>> accum;

        REQUIRE(accum.get_sums() == vst{0, 0});
        accum.accept(2);
        REQUIRE(accum.get_sums() == vst{0, 1});
        accum.accept(3);
        REQUIRE(accum.get_sums() == vst{1, 2});

        accum.accept(2);
        try {
            accum.accept(2);
            FAIL("should have thrown on overflow");
        } catch (std::runtime_error &) {
        }
    }


}






