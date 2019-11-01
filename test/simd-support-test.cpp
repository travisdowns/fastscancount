#include "simd-support.hpp"

#include <numeric>

// include me last
#include "catch.hpp"

using vu8  = std::vector<uint8_t>;
using vu16 = std::vector<uint16_t>;
using vu32 = std::vector<uint32_t>;
using vu64 = std::vector<uint64_t>;

int array[100];

struct FillArray {
    FillArray() { std::iota(std::begin(array), std::end(array), 0); }
};

FillArray dummy;

TEST_CASE("dummy") {
    CHECK(array[10] == 10);
}

template <size_t W = 8>
vu32 scalar_pack(const vu32& values, const vu32& mask) {
    CHECK(values.size() == W);
    CHECK(mask.size() == W);

    vu32 ret;
    for (size_t i = 0; i < W; i++) {
        if (mask[i]) {
            ret.push_back(values[i]);
        }
    }
    ret.resize(W);
    return ret;
}

TEST_CASE("to_vector") {
    CHECK(to_vector(to_simd<__m256i>(vu32{1, 2, 3, 4, 5, 6, 7, 8})) == vu32{1, 2, 3, 4, 5, 6, 7, 8});
}


#ifdef __AVX512F__
TEST_CASE("to_vector-512") {
    __m512i vec = to_simd<__m512i>(vu64{1, 2, 3, 4, 5, 6, 7, 8});
    CHECK(to_vector<uint64_t>(vec) == vu64{1, 2, 3, 4, 5, 6, 7, 8});
    CHECK(to_vector<uint32_t>(vec) == vu32{1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0});
}
#endif


vu32 frog_pack_adapt(const vu32& values, const vu32& mask) {
    return to_vector(pack_left_epi32(to_simd<__m256i>(values), to_simd<__m256i>(mask)));
}

template <typename F>
void compare(F f, const vu32& values, const vu32& mask) {
    size_t count = std::count_if(mask.begin(), mask.end(), [](auto e) { return e; });  // count set elements
    auto r       = f(values, mask);
    auto e       = scalar_pack(values, mask);
    r.resize(count);
    e.resize(count);
    CHECK(r == e);
}

TEST_CASE("pack-left") {
    uint32_t x = -1;  // x is an "on" mask

    vu32 v = {1, 2, 3, 4, 5, 6, 7, 8};
    vu32 m = {0, 0, 0, 0, 0, 0, 0, 0};

    SECTION("scalar_pack") {
        CHECK(scalar_pack(v, {0, 0, 0, 0, 0, 0, 0, 0}) == vu32{0, 0, 0, 0, 0, 0, 0, 0});
        CHECK(scalar_pack(v, {x, 0, 0, 0, 0, 0, 0, 0}) == vu32{1, 0, 0, 0, 0, 0, 0, 0});
        CHECK(scalar_pack(v, {x, 0, 0, 0, 0, 0, 0, x}) == vu32{1, 8, 0, 0, 0, 0, 0, 0});
        CHECK(scalar_pack(v, {0, 0, 0, 0, 0, 0, 0, x}) == vu32{8, 0, 0, 0, 0, 0, 0, 0});
        CHECK(scalar_pack(v, {x, x, x, x, x, x, x, x}) == vu32{1, 2, 3, 4, 5, 6, 7, 8});
    }

    SECTION("frog_pack") {
        compare(frog_pack_adapt, v, {0, 0, 0, 0, 0, 0, 0, 0});
        compare(frog_pack_adapt, v, {x, 0, 0, 0, 0, 0, 0, 0});
        compare(frog_pack_adapt, v, {x, 0, 0, 0, 0, 0, 0, x});
        compare(frog_pack_adapt, v, {0, 0, 0, 0, 0, 0, 0, x});
        compare(frog_pack_adapt, v, {x, x, x, x, x, x, x, x});
    }
}

template <typename T = uint32_t>
std::vector<T> iota_vec(T start = 0) {
    std::vector<T> ret(32 / sizeof(T));
    std::iota(ret.begin(), ret.end(), start);
    return ret;
}

TEST_CASE("iota_vec") {
    CHECK(iota_vec<uint8_t>() == vu8{0, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    CHECK(iota_vec<uint16_t>() == vu16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    CHECK(iota_vec<uint32_t>() == vu32{0, 1, 2, 3, 4, 5, 6, 7});
}


vu8 sll_ref(const vu8& v, int n) {
    vu8 ret(32);
    REQUIRE(n >= 0);
    REQUIRE(n <= 32);
    std::copy_n(v.begin(), 32 - n, ret.begin() + n);
    return ret;
}

template <int N>
void check_sll(const vu8& v) {
    INFO("N = " << N);
    auto ref = sll_ref(v, N);
    auto simd = to_vector<uint8_t>(ss_mm256_sllx_si256<N>(to_simd<__m256i>(v)));;
    REQUIRE(simd == ref);
}

template<int... N>
void check_all_sll(const vu8& v, std::integer_sequence<int, N...>)
{
    (void)std::initializer_list<int>{(check_sll<N>(v), 0)...};
}

TEST_CASE("ss_mm256_sllx_si256") {
    vu8 v = iota_vec<uint8_t>();
    check_all_sll(v, std::make_integer_sequence<int, 32>());
}
