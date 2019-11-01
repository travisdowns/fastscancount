/*
 * unit-test.cpp
 */

#include "fastbitset.hpp"

#include <bitset>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <random>
#include <type_traits>
#include <vector>
#include <optional>

template <typename T>
struct dumb_list {
    using value_type = T;
    size_t size() const { return 0; }
    void push_back(T t) { if (t > 0) elems.push_back(t); }

    auto begin() const { return elems.begin(); }
    auto end() const { return elems.end(); }

    std::vector<T> elems;
};

template<std::size_t N>
std::string to_string(const std::bitset<N> &in) {
    std::string str = in.to_string();
    std::reverse(str.begin(), str.end());
    return str;
}

template <size_t N>
std::ostream& operator<<(std::ostream& os, const std::bitset<N>& bitset) {
    os << to_string(bitset);
    return os;
}

template <size_t N>
std::ostream& operator<<(std::ostream& os, const fastbitset<N>& bitset) {
    os << to_string(bitset);
    return os;
}

#include "catch.hpp"

constexpr size_t bit_count = 128;

using elem_type = int;
using left_t  = std::bitset<bit_count>;
using right_t = fastbitset<bit_count>;


struct mc_result {
    bool same_result;
    std::string left, right;
};

template <typename F, typename T>
auto remapping_call(F f, T& t, size_t arg) {
    using ret_type = decltype(f(t, arg));
    if constexpr (std::is_same<ret_type,void>::value) {
        f(t, arg);
        return std::optional<int>();
    } else {
        return std::optional(f(t, arg));
    }
}

template <typename L, typename R>
mc_result mutate_and_check(L opl, R opr, left_t& l, right_t& r, size_t arg) {
    using namespace std; // for to_string

    auto lresult = remapping_call(opl, l, arg);
    auto rresult = remapping_call(opr, r, arg);
    if (lresult && rresult) {
        if (!(lresult.value() == rresult.value())) {
            return { false, to_string(lresult.value()), to_string(rresult.value()) };
        }
    }
    return { true , "", "" };
}

struct op_def {
    std::string name;

    std::function<mc_result(left_t& l, right_t& r, size_t)> mc_op;

    template <typename F>
    op_def(std::string name, F op) : op_def(name, op, op) {}

    template <typename L, typename R>
    op_def(std::string name, L lop, R rop) : name{std::move(name)} {
        mc_op = [lop, rop](left_t& l, right_t& r, size_t arg){ return mutate_and_check<L,R>(lop, rop, l, r, arg); };
    }
};

template <typename T>
size_t to_ulong(const T& t) {
    try {
        return t.to_ulong();
    } catch (std::overflow_error&) {
        return -1;
    }
}

template <typename T>
size_t find_next(const T& t, size_t pos) {
    for (size_t i = pos + 1; i < t.size(); i++) {
        if (t.test(i)) {
            return i;
        }
    }
    return -1;
}

std::vector<op_def> ops = {
    { "size()",     [](auto& v, size_t st_arg) { return v.size(); } },
    { "set(%zu)",   [](auto& v, size_t st_arg) { v.set(st_arg); } },
    { "reset(%zu)", [](auto& v, size_t st_arg) { v.reset(st_arg); } },
    { "test(%zu)",  [](auto& v, size_t st_arg) { return v.test(st_arg); } },
    { "count()",    [](auto& v, size_t st_arg) { return v.count(); } },
    { "any()",      [](auto& v, size_t st_arg) { return v.any(); } },
    { "to_ulong()", [](auto& v, size_t st_arg) { return to_ulong(v); } },
    { "flip(%zu)",  [](auto& v, size_t st_arg) { return v.flip(st_arg); } },
    { "find_first()",   [](auto& v, size_t st_arg) { return find_next(v, -1); }, [](auto& v, size_t st_arg) { return v.find_first(); } },
    { "find_next(%zu)", [](auto& v, size_t st_arg) { return find_next(v, st_arg); }, [](auto& v, size_t st_arg) { return v.find_next(st_arg); } },
#define SUBSET(from,to) { "subset(" #from "," #to ")", \
    [](auto& v, size_t st_arg) { return bitsubset<to - from>(v, from, to); }, \
    [](auto& v, size_t st_arg) { return v.template subset<to - from>(from); } }
    SUBSET(0, 64),
    SUBSET(64, 128),
    SUBSET(0, 128),
};

template <size_t M, size_t N>
bool operator==(const std::bitset<M>& l, const fastbitset<N>& r) {
    if (l.size() != r.size()) {
        return false;
    }
    for (size_t i = 0; i < l.size(); i++) {
        if (l.test(i) != r.test(i)) {
            return false;
        }
    }
    return true;
}

bool compare(const left_t& l, const right_t& r) {
    // if (!std::equal(l.begin(), l.end(), r.begin(), r.end())) {
    if (!(l == r)) {
        WARN("left :" << l << "\nright:" << r);
        return false;
    } else {
        REQUIRE(l == r); // increment assertion count

    }
    return true;
}

/*
 * Given a printf-style format and args, return the formatted string as a std::string.
 *
 * See https://stackoverflow.com/a/26221725/149138.
 */
template<typename ... Args>
std::string string_format(const std::string& format, Args ... args) {
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

/* stricty speaking this overload is not necessary but it avoids a gcc warning about a format operation without args */
static inline std::string string_format(const std::string& format) {
    return format;
}



TEST_CASE( "basic" ) {
    // fastbitset<128> bs;
    // bs.subset<0,64>();
}

TEST_CASE( "random-ops" ) {

    std::mt19937_64 rng;
    std::uniform_int_distribution<size_t> op_dist(0, ops.size() - 1);
    std::uniform_int_distribution<size_t> arg_dist(0, bit_count - 1);

    for (size_t outer = 0; outer < 100; outer++) {

        left_t l;
        right_t r;

        size_t iters = 10;
        std::string all_ops_string;

        if (!compare(l, r)) {
            FAIL("not equal initially");
        }

        for (size_t inner = 0; inner < 10000; inner++) {

            // select an op
            auto op_idx = op_dist(rng);
            auto& op = ops.at(op_idx);
            auto arg = arg_dist(rng);
            std::string format = std::string("op %zu: ") + op.name + "\n";
            auto desc = string_format(format, inner, arg);
            // printf("%s", desc.c_str());
            all_ops_string += desc;

            auto result = op.mc_op(l, r, arg);

            if (!result.same_result) {
                INFO("left :" << l << "\nright:" << r << "\ncall: " << desc << "\nleft  result: " << result.left << "\nright result: " << result.right);
                FAIL("method results differed, ops:\n" << all_ops_string);
            }
            REQUIRE(result.same_result);

            if (!compare(l, r)) {
                // CHECK(false);
                FAIL("not equal after these ops:\n" << all_ops_string);
            }
        }
    }


}


