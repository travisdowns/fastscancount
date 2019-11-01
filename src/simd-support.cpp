#include "simd-support.hpp"

#include <cstring>
#include <ostream>

uint8_t g_pack_left_table_uint8_tx3[256 * 3 + 1];

u_int32_t get_nth_bits(int a) {
    u_int32_t out = 0;
    int c = 0;
    for (int i = 0; i < 8; ++i) {
        auto set = (a >> i) & 1;
        if (set) {
            out |= (i << (c * 3));
            c++;
        }
    }
    return out;
}

struct BuildPackMask {
    BuildPackMask() {
        for (int i = 0; i < 256; ++i) {
            uint32_t bits = get_nth_bits(i);
            std::memcpy(g_pack_left_table_uint8_tx3 + i * 3, &bits, sizeof(uint32_t));
        }
    }
};

std::ostream& operator<<(std::ostream& os, __m256i v) {
    auto vec = to_vector(v);
    os << '[';
    bool first = true;
    for (auto x : vec) {
        if (!first) os << ", ";
        os << x;
        first = false;
    }
    os << ']';
    return os;
}

BuildPackMask builder;

