#include <algorithm>
#include <assert.h>
#include <iostream>
#include <numeric>
#include <tuple>
#include <iterator>

#include "analyze.hpp"
#include "common.h"

const size_t chunk_size = 40000;


struct id_count {
    uint32_t id;
    uint32_t count;
    bool operator<(const id_count& rhs) const {
        return std::tie(count, id) < std::tie(rhs.count, rhs.id);
    }
};

std::ostream& operator<<(std::ostream& os, id_count idc) {
    os << "id: " << idc.id << " count: " << idc.count;
    return os;
}

template <typename C>
std::vector<size_t> size_array(const C& container) {
    std::vector<size_t> ret(container.size());
    std::transform(container.begin(), container.end(), ret.begin(), [](auto& d) {return d.size();});
    return ret;
}

template <typename T>
class histogram {

    std::vector<T> bounds;
    std::vector<size_t> counts;
    T sum;

public:
    histogram(std::vector<T> bounds) : bounds{std::move(bounds)}, counts(this->bounds.size() + 1), sum{0} {
        assert(!bounds.empty());
        assert(std::is_sorted(bounds.begin(), bounds.end()));
    }

    void accept(const T& t) {
        size_t b = 0;
        for (; b < bounds.size(); b++) {
            if (t < bounds[b])
                break;
        }
        ++counts.at(b);
        sum += t;
    }

    void out(std::ostream& os) {
        os << "Average: " << ((double)sum / std::accumulate(counts.begin(), counts.end(), 0)) << "\n";
        os << "< " << bounds.front() << ": " << counts.front() << "\n";
        for (size_t b = 0; b + 1 < bounds.size(); b++) {
            os << "[" << bounds.at(b) << "," << bounds.at(b + 1) << "): " << counts.at(b + 1) << "\n";
        }
        os << "> " << bounds.back() << ": " << counts.back() << "\n";
    }

};

void check_chunk_freq(const std::vector<uint32_t>& array, size_t chunk) {

    size_t i = 0, upper_bound = chunk;

    histogram<size_t> histo({0,1,2,3,4,5,6,7,9,10});

    while (i < array.size()) {
        size_t count = 0;
        while (i < array.size() && array.at(i) < upper_bound) {
            count++;
            i++;
        }
        histo.accept(count);
        // printf("Count for %zu to %zu : %zu\n", upper_bound - chunk, upper_bound - 1, count);
        upper_bound += chunk;
    }

    histo.out(std::cout);

}

void analyze(const std::vector<std::vector<uint32_t>>& data,
             const std::vector<std::vector<uint32_t>>& queries)
{
    // check out how uniform (or not) the data is
    uint32_t largest = get_largest(data);
    std::cout << "Largest max : " << largest << "\nSmallest max: " << get_smallest_max(data) << "\n";

    auto data_sizes = size_array(data);
    std::sort(data_sizes.begin(), data_sizes.end());
    size_t total_dsize = std::accumulate(data_sizes.begin(), data_sizes.end(), 0);
    // for (auto s : data_sizes) {
    //     std::cout << "data size: " << s << "\n";
    // }
    std::cout << "Total posting IDs: " << total_dsize << "\n";
    std::cout << "Min posting size: " << *std::min_element(data_sizes.begin(), data_sizes.end()) << "\n";
    std::cout << "Avg query size  : " << total_dsize / data.size() << "\n";
    std::cout << "Max posting size: " << *std::max_element(data_sizes.begin(), data_sizes.end()) << "\n";

    auto query_sizes = size_array(queries);
    std::sort(query_sizes.begin(), query_sizes.end());
    size_t total_qsize = std::accumulate(query_sizes.begin(), query_sizes.end(), 0);
    std::cout << "Min query size  : " << *std::min_element(query_sizes.begin(), query_sizes.end()) << "\n";
    std::cout << "Avg query size  : " << total_qsize / queries.size() << "\n";
    std::cout << "Max query size  : " << *std::max_element(query_sizes.begin(), query_sizes.end()) << "\n";

    std::cout << "Overall ID density (IDs per ID domain): " << (double)total_dsize / (largest + 1) << "\n";
    std::cout << "Average per-list ID density           : " << ((double)total_dsize / data.size()) / (largest + 1) << "\n";
    std::cout << "Average per-query ID density          : " << ((double)total_qsize / queries.size() * total_dsize / data.size())
            / (largest + 1) << "\n";

    // std::cou    t << std::flush;

    size_t totali = 0;
    // calculate element frequncy
    std::vector<uint32_t> counts(largest + 1);
    for (auto& d : data) {
        // std::copy_n(d.begin(), 10, std::ostream_iterator<uint32_t>(std::cout, " "));
        // std::cout << "\n";
        for (auto e : d) {
            ++counts.at(e);
            // std::cout << "X: " << counts.at(e) << "\n";
            totali++;
        }

        check_chunk_freq(d, 512);
    }

    std::cout << "totali: " << totali << "\n";

    std::vector<id_count> sorted_counts(counts.size());
    for (uint32_t i = 0; i < counts.size(); i++) {
        sorted_counts.at(i) = {i, counts.at(i)};
    }

    std::sort(sorted_counts.begin(), sorted_counts.end());

    std::copy_n(sorted_counts.begin(), 100, std::ostream_iterator<id_count>(std::cout, "\n"));
    std::cout << "----------------\n";
    std::copy_n(sorted_counts.end() - 100, 100, std::ostream_iterator<id_count>(std::cout, "\n"));


}

// void analyze_scancount(const std::vector<const std::vector<uint32_t>*> &data,
//                        std::vector<uint32_t> &out, size_t threshold) {
//   uint64_t largest = 0;
//   for(auto z : data) {
//     const std::vector<uint32_t> & v = *z;
//     if(v[v.size() - 1] > largest) largest = v[v.size() - 1];
//   }
//   std::vector<uint8_t> counters(largest+1);
//   out.clear();
//   for (size_t c = 0; c < data.size(); c++) {
//     const std::vector<uint32_t> &v = *data[c];
//     for (size_t i = 0; i < v.size(); i++) {
//       counters[v[i]]++;
//     }
//   }
//   for (uint32_t i = 0; i < counters.size(); i++) {
//     if (counters[i] > threshold)
//       out.push_back(i);
//   }
// }
