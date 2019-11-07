#ifndef ACCUM7_H_
#define ACCUM7_H_

#include "compressed-bitmap.hpp"

#include <array>
#include <vector>

namespace fastscancount {

template <typename T, typename B>
struct traits_base {

    struct carry_sum {
        T carry;
        T sum;
    };

        /* aka half adder */
    static carry_sum add2(T a, T b) {
        return {B::and_(a, b), B::xor_(a, b)};
    }

    /* aka full adder */
    static carry_sum add3(T a, T b, T c) {
        auto xor01 = B::xor_(a, b);
        return {
            B::or_(B::and_(a, b), B::and_(c, xor01)), // carry
            B::xor_(xor01, c) };                // sum
    }
};

template <typename T>
struct default_traits : traits_base<T, default_traits<T>> {

    static T xor_(const T& l, const T& r) {
        return l ^ r;
    }

    static T and_(const T& l, const T& r) {
        return l & r;
    }

    static T or_(const T& l, const T& r) {
        return l | r;
    }

    static T not_(const T& v) {
        return ~v;
    }

    static bool test(const T& v, size_t idx) {
        return v[idx];
    }
};

/**
 * Accumulator that can accumulate per-bit sums up to B bits. When a
 * value reaches the max of 2^B, it saturates.
 */
template <size_t B, typename T, typename traits>
class accumulator {

    std::array<T,B> bits;
    T sat;

public:
    static constexpr size_t max = (size_t)1 << B;

    accumulator(size_t initial = 0) : bits{}, sat{} {
        T ones = traits::not_(T{});
        while (initial--) {
            accept(ones);
        }
    }

    void accept(T addend) {
        T carry = addend;
        for (size_t bit = 0; bit < B; bit++) {
            T sum = traits::xor_(bits[bit], carry);
            carry = traits::and_(bits[bit], carry);
            bits[bit] = sum;
        }
        sat = traits::or_(sat, carry);  // update saturation
    }

    void accept7(T v0, T v1, T v2, T v3, T v4, T v5, T v6) {
        auto [c0_0, s0_0] = traits::add3(v0, v1, v2);
        auto [c0_1, s0_1] = traits::add3(v3, v4, v5);
        auto [c0_2, s0_2] = traits::add3(v6, s0_0, s0_1);

        auto [c1_0, s1_0] = traits::add3(c0_0, c0_1, c0_2);

        // output is s0_2, s1_0, c1_0 with weight 0, 1, 2
        accept_weighted(s0_2, s1_0, c1_0);
    }

    /**
     * Propagate carry starting from position P.
     */
    template <size_t P, size_t S>
    void accept_weighted2(T carry_in, const std::array<T,S>& values) {
        if constexpr (P < B) {
            if constexpr (P < S) {
                auto [c, s] = traits::add3(carry_in, values[P], bits[P]);
                bits[P] = s;
                accept_weighted2<P+1,S>(c, values);
            } else {
                auto [c, s] = traits::add2(carry_in, bits[P]);
                bits[P] = s;
                accept_weighted2<P+1,S>(c, values);
            }
        } else {
            sat = traits::or_(sat, carry_in);
            if constexpr (P < S) {
                sat = traits::or_(sat, values[P]);
                accept_weighted2<P+1,S>(T{}, values);
            }
        }
    }

    /**
     * Accept v0, v1, v2 with weights 2^0, 2^1, 2^2
     * respecitvely.
     */
    void accept_weighted(T v0, T v1, T v2) {
        static_assert(B >= 1);

        auto [c0, s0] = traits::add2(v0, bits[0]);
        bits[0] = s0;

        accept_weighted2<1, 3>(c0, {v0, v1, v2});
    }

    /**
     * A vector with one element per vertical counter containing its sum.
     *
     * This method is very slow and is intended primarily for writing tests.
     */
    std::vector<size_t> get_sums() {
        std::vector<size_t> ret(traits::size());
        for (size_t i = 0; i < traits::size(); i++) {
            size_t multiplier = 1;
            for (size_t bit = 0; bit < bits.size(); bit++) {
                ret[i] += traits::test(bits[bit], i) * multiplier;
                multiplier *= 2;
            }
            if (traits::test(sat, i)) {
                ret[i] = max; // saturation
            }
        }
        return ret;
    }

    T get_saturated() {
        return sat;
    }
};

}

#endif
