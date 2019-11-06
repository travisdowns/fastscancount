#ifndef ACCUM7_H_
#define ACCUM7_H_

#include "bitscan.hpp"

using fastscancount::default_traits;

template <typename T, typename traits = default_traits<T>>
class accum7 {
    T bits0, bits1, bits2, sat;

    std::array<T*, 3> as_array() {
        return {&bits0, &bits1, &bits2};
    }

public:

    static constexpr size_t max = 8;

    accum7(size_t initial = 0) : bits0{}, bits1{}, bits2{}, sat{} {
        T ones = traits::not_(T{});
        while (initial--) {
            accept(ones);
        }
    }

    void accept(T addend) {
        T carry;

        carry = traits::and_(bits0, addend);
        bits0  =  traits::xor_(bits0, addend);
        addend = carry;

        carry = traits::and_(bits1, addend);
        bits1  =  traits::xor_(bits1, addend);
        addend = carry;

        carry = traits::and_(bits2, addend);
        bits2  =  traits::xor_(bits2, addend);
        addend = carry;

        sat = traits::or_(sat, addend);
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
     * Accept v0, v1, v2 with weights 2^0, 2^1, 2^2,
     * respecitvely.
     */
    void accept_weighted(T v0, T v1, T v2) {
        auto [c0, s0] = traits::add2(v0, bits0);
        bits0 = s0;

        auto [c1, s1] = traits::add3(c0, v1, bits1);
        bits1 = s1;

        auto [c2, s2] = traits::add3(c1, v2, bits2);
        bits2 = s2;

        sat = traits::or_(sat, c2);
    }

    /**
     * A vector with one element per vertical counter containing its sum.
     *
     * This method is very slow and is intended primarily for writing tests.
     */
    std::vector<size_t> get_sums() {
        std::vector<size_t> ret(traits::size());
        auto bits = as_array();
        for (size_t i = 0; i < traits::size(); i++) {
            size_t multiplier = 1;
            for (size_t bit = 0; bit < bits.size(); bit++) {
                ret[i] += traits::test(*bits[bit], i) * multiplier;
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

#endif
