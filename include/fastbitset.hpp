#ifndef FAST_BITSET_H_
#define FAST_BITSET_H_

#include <algorithm>
#include <array>
#include <stdlib.h>
#include <inttypes.h>
#include <limits.h>
#include <string>
#include <assert.h>


template <size_t N>
class fastbitset {
    using block_t = uint32_t;
    static const size_t block_bits = sizeof(block_t) * CHAR_BIT;
    static const size_t block_count = (N + block_bits - 1) / block_bits;

    static_assert(N % block_bits == 0, "size must be a multiple of 64");
    static_assert(sizeof(block_bits) <= sizeof(unsigned long), "builtins need this");

    std::array<block_t, block_count> blocks;

    template <size_t M>
    friend class fastbitset;

public:

    static constexpr size_t npos = -1;

    fastbitset() : blocks{} {}

    void set(std::size_t pos, bool value = true) {
        if (value) {
            block_at(pos) |= mask_for(pos);
         } else {
            block_at(pos) &= ~mask_for(pos);
         }
    }

    void reset(std::size_t pos) {
        set(pos, false);
    }

    void flip(std::size_t pos) {
        block_at(pos) ^= mask_for(pos);
    }

    bool test(size_t pos) const {
        return block_at(pos) & mask_for(pos);
    }

    size_t find_first() const {
        return find_next(-1);
    }

    size_t find_next(size_t pos) const {
        for (size_t i = pos + 1; i < size(); i++) {
            if (test(i)) {
                return i;
            }
        }
        return npos;
    }

    size_t size() const {
        return N;
    }

    size_t count() const {
        size_t c = 0;
        for (auto b : blocks) {
            c += popcount(b);
        }
        return c;
    }

    bool any() {
        return count();
    }

    unsigned long to_ulong() const {
        static_assert(sizeof(unsigned long) % sizeof(block_t) == 0);
        size_t i = 0;
        unsigned long ret = 0;
        for (; i < std::min(sizeof(unsigned long) / sizeof(block_t), blocks.size()); i++) {
            ret |= (((unsigned long)blocks[i]) << i * block_bits);
        }
        for (; i < blocks.size(); i++) {
            if (popcount(blocks[i])) {
                throw std::overflow_error("to_ulong can't represent");
            }
        }
        return ret;
    }

    template <size_t NEW>
    fastbitset<NEW> subset(size_t from) {
        assert(from % block_bits == 0);
        fastbitset<NEW> ret;
        std::copy_n(blocks.begin() + pos_to_bidx(from), ret.size() / block_bits, ret.blocks.begin());
        return ret;
    }

    fastbitset<N> operator^(const fastbitset& rhs) const {
        fastbitset<N> ret(*this);
        ret.apply(rhs, [](block_t& l, block_t r) { l ^= r; });
        return ret;
    }

    fastbitset<N> operator&(const fastbitset& rhs) const {
        fastbitset<N> ret(*this);
        ret.apply(rhs, [](block_t& l, block_t r) { l &= r; });
        return ret;
    }

    fastbitset<N> operator|(const fastbitset& rhs) const {
        fastbitset<N> ret(*this);
        ret.apply(rhs, [](block_t& l, block_t r) { l |= r; });
        return ret;
    }

    fastbitset<N> operator~() const {
        fastbitset<N> ret(*this);
        ret.apply([](block_t& l) { l = ~l; });
        return ret;
    }

private:

    static size_t popcount(block_t b) {
        return __builtin_popcountl(b);
    }

    static size_t tzcnt(block_t b) {
        assert(b);
        return __builtin_ctzl(b);
    }

    static constexpr size_t pos_to_bidx(size_t pos) {
        return pos / block_bits;
    }

    static void check_index(size_t pos) {
        assert(pos / block_bits < block_count);
    }

    template <typename OP>
    void apply(OP op) {
        for (auto& b : blocks) {
            op(b);
        }
    }

    template <typename OP, size_t NN>
    static void apply(fastbitset<NN>& left, fastbitset<NN>& right, OP op) {
        for (size_t i = 0; i < left.block_count; i++) {
            op(left.blocks[i], right.blocks[i]);
        }
    }

    template <typename OP>
    void apply(const fastbitset<N>& right, OP op) {
        for (size_t i = 0; i < block_count; i++) {
            op(this->blocks[i], right.blocks[i]);
        }
    }

    const block_t& block_at(size_t pos) const {
        check_index(pos);
        return blocks[pos_to_bidx(pos)];
    }

    block_t& block_at(size_t pos) {
        check_index(pos);
        return blocks[pos_to_bidx(pos)];
    }

    static block_t mask_for(size_t pos) {
        check_index(pos);
        return ((block_t)1) << (pos % block_bits);
    }
};

template<std::size_t N>
std::string to_string(const fastbitset<N> &in) {
    std::string str;
    for (size_t i = 0; i < in.size(); i++) {
        str += in.test(i) ? "1" : "0";
    }
    return str;
}

template <size_t M, template<size_t> class T, size_t N>
T<M> bitsubset(const T<N>& in, size_t from, size_t to) {
    assert(to - from == M);
    assert(to <= in.size());
    assert(from <= to);
    T<M> ret;
    for (size_t i = 0; i < (to - from); i++) {
        ret.set(i, in.test(i + from));
    }
    return ret;
}

#endif
