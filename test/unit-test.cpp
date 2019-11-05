/*
 * unit-test.cpp
 */

#include "accum7.hpp"
#include "bitscan.hpp"
#include "catch.hpp"
#include "compressed-bitmap.hpp"

#include <random>

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

    template <size_t C>
    static auto make7(size_t initial = 0) {
        return accum7<int, int_traits<C>>(initial);
    }
};

TEST_CASE("accumulator") {
    accum_test<int_holder>();
}

#ifdef __AVX512F__

/**
 * wrapper for accumulator<B, __m512i, m512_traits> so that
 * accept takes a plain integer like the version above, so
 * the tests can be shared.
 *
 * The counter is limited to C counters for sanity.
 */
template <size_t C, typename base>
struct accum512 : base {
    static_assert(C <= 32, "C too big"); // otherwise stuff that returns uint32_t won't work

    accum512(size_t initial) : base{initial} {}

    __m512i to_vec(int i) {
        return _mm512_set_epi32(
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, i
        );
    }

    void accept(int v) {
        base::accept(to_vec(v));
    }

    void accept7(int v0, int v1, int v2, int v3, int v4, int v5, int v6) {
        base::accept7(to_vec(v0), to_vec(v1), to_vec(v2), to_vec(v3), to_vec(v4), to_vec(v5), to_vec(v6));
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
        return accum512<C, fastscancount::accumulator<B, __m512i, fastscancount::m512_traits>>(initial);
    }

    template <size_t C>
    static auto make7(size_t initial = 0) {
        return accum512<C, accum7<__m512i, fastscancount::m512_traits>>(initial);;
    }

};


TEST_CASE("accumulator-512") {
    accum_test<accum512_holder>();
}
#endif

template <typename A>
void accept_n(A& a, int val, size_t n) {
    while (n--) {
        a.accept(val);
    }
}

template <typename T>
void test_accum7() {
    {
        auto accum = T::template make7<1>();

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
        REQUIRE(accum.get_sums() == vst{5});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{6});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{7});
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{8});  // saturated
        accum.accept(1);
        REQUIRE(accum.get_sums() == vst{8});  // saturated
        accum.accept(0);
        REQUIRE(accum.get_sums() == vst{8});  // saturated

    }

    {
        auto accum = T::template make7<2>();

        REQUIRE(accum.get_sums() == vst{0, 0});
        accum.accept(2);
        for (size_t i = 0; i < 10; i++) {
            size_t sum0 = std::min(accum.max, i + 1);
            size_t sum1 = std::min(accum.max, i);

            REQUIRE(accum.get_sums() == vst{sum1, sum0});

            accum.accept(3);
        }

    }

    {
        auto accum = T::template make7<2>();

        accum = T::template make7<2>(1);
        REQUIRE(accum.get_sums() == vst{1, 1});

        accum = T::template make7<2>(2);
        REQUIRE(accum.get_sums() == vst{2, 2});

        accum = T::template make7<2>(3);
        REQUIRE(accum.get_sums() == vst{3, 3});

        accum = T::template make7<2>(10);
        REQUIRE(accum.get_sums() == vst{8, 8});
    }

    {
        auto accum = T::template make7<2>();

        REQUIRE(accum.get_saturated() == 0b00);

        accum.accept(0b10);
        REQUIRE(accum.get_saturated() == 0b00);
        accum.accept(0b11); // 2 1
        REQUIRE(accum.get_saturated() == 0b00);
        accum.accept(0b11); // 3 2
        REQUIRE(accum.get_saturated() == 0b00);
        accum.accept(0b11); // 4 3
        REQUIRE(accum.get_saturated() == 0b00);

        accept_n(accum, 0b11, 4);
        REQUIRE(accum.get_saturated() == 0b10);
        accum.accept(0b01);
        REQUIRE(accum.get_saturated() == 0b11);
        accept_n(accum, 0b11, 10);
        REQUIRE(accum.get_saturated() == 0b11);
    }

    std::mt19937_64 gen(0x12345678);
    std::bernoulli_distribution d(0.5);

    for (int iter = 0; iter < 10; iter++) {
        auto accum = T::template make7<2>();
        size_t sum0 = 0, sum1 = 0;
        for (int inner = 0; inner < 20; inner++) {

            REQUIRE(accum.get_sums() == vst{sum0, sum1});

            int add0 = d(gen);
            int add1 = d(gen);

            accum.accept(add0 | (add1 << 1));

            sum0 = std::min(accum.max, sum0 + add0);
            sum1 = std::min(accum.max, sum1 + add1);
        }
    }

    for (int iter = 0; iter < 10; iter++) {
        auto accum = T::template make7<2>();
        size_t sum0 = 0, sum1 = 0;
        for (int inner = 0; inner < 20; inner++) {

            REQUIRE(accum.get_sums() == vst{sum0, sum1});

            bool is_sum1 = d(gen);

            int add0 = d(gen) << is_sum1;
            int add1 = d(gen) << is_sum1;
            int add2 = d(gen) << is_sum1;
            int add3 = d(gen) << is_sum1;
            int add4 = d(gen) << is_sum1;
            int add5 = d(gen) << is_sum1;
            int add6 = d(gen) << is_sum1;

            accum.accept7(add0, add1, add2, add3, add4, add5, add6);

            auto& sum = is_sum1 ? sum1 : sum0;
            sum = std::min(accum.max,
                    sum + (add0 + add1 + add2 + add3 + add4 + add5 + add6 >> (is_sum1 ? 1 : 0)));
        }
    }

}

TEST_CASE("accum7")
{
    test_accum7<int_holder>();
}

TEST_CASE("accum7-512")
{
    test_accum7<accum512_holder>();
}
