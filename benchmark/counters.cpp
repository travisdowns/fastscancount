// Fine-grained statistics is available only on Linux
#include "fastscancount.h"
#include "ztimer.h"
#include "analyze.hpp"
#include "bitscan.hpp"
#ifdef __AVX2__
#include "fastscancount_avx2.h"
#include "fastscancount_avx2b.h"
#endif
#ifdef __AVX512F__
#include "fastscancount_avx512.h"
#endif
#include "linux-perf-events-wrapper.h"
#include "maropuparser.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <stdexcept>

/////////////////////////////
// demo_random mode params //
/////////////////////////////

/* the number of times to run each benchmark in demo mode */
#define REPEATS 1
#define START_THRESHOLD  3
#define END_THRESHOLD   11

// constexpr size_t DEMO_DOMAIN  = 20000000;
// constexpr size_t DEMO_ARRAY_SIZE  = 50000;
// constexpr size_t DEMO_ARRAY_COUNT =   100;

// fast params
constexpr size_t DEMO_DOMAIN  =   200000;
constexpr size_t DEMO_ARRAY_SIZE  = 5000;
constexpr size_t DEMO_ARRAY_COUNT =   10;

//////////////////////
// data mode params //
//////////////////////

/* the maximum number of queries to read from queries.bin */
constexpr size_t MAX_QUERIES = 1000; // 100;

constexpr bool PRINT_ALL = true;
constexpr bool do_analyze = false;

/////////////////////
// all mode params //
/////////////////////
#define RUNNINGTESTS

constexpr int NAME_WIDTH = 25;
constexpr int  COL_WIDTH = 16;

// extra perf events
constexpr int EVENT_L1D_REPL          =     0x151;
constexpr int EVENT_UOPS_ISSUED       =     0x10e;
constexpr int EVENT_PEND_MISS         =     0x148;
constexpr int EVENT_PEND_MISS_CYCLES  = 0x1000148;


#if defined(__linux__) && !defined(NO_COUNTERS)
#define USE_COUNTERS
#endif

/* a function that returns the normalized event value given the value, cycles and sum */
using raw_extractor = double (double val, double cycles, size_t total_elems);

#define HW_EVENT(suffix) {PERF_TYPE_HARDWARE, PERF_COUNT_HW_ ## suffix }

struct column_spec {
  TypeAndConfig event;
  const char* desc;
  raw_extractor* extractor;

  double get_result(LinuxEventsWrapper& w, size_t total_elems) {
    double cycles = w.get_result(HW_EVENT(CPU_CYCLES));
    return extractor(w.get_result(event), cycles, total_elems);
  }
};

std::vector<column_spec> all_columns = {
#ifdef USE_COUNTERS
  // you must always leave CPU_CYCLES enabled
  { HW_EVENT(CPU_CYCLES   ),                  "cycles/element", [](double val, double cycles, size_t sum) -> double { return val / sum;          } },
  // { HW_EVENT(INSTRUCTIONS ),                  "instr/cycle",    [](double val, double cycles, size_t sum) -> double { return val / cycles; } },
  // { HW_EVENT(INSTRUCTIONS ),                  "instr/elem",     [](double val, double cycles, size_t sum) -> double { return val / sum; } },
  // { HW_EVENT(BRANCH_MISSES),                  "miss/element",   [](double val, double cycles, size_t sum) -> double { return val / sum;          } },
  { {PERF_TYPE_RAW, EVENT_L1D_REPL,        }, "l1 repl/element",[](double val, double cycles, size_t sum) -> double { return val / sum;    } },
  { {PERF_TYPE_RAW, EVENT_UOPS_ISSUED,     }, "uops/cycle",     [](double val, double cycles, size_t sum) -> double { return val / cycles; } },
  // { {PERF_TYPE_RAW, EVENT_UOPS_ISSUED,     }, "uops/elem",      [](double val, double cycles, size_t sum) -> double { return val / sum; } },
  // { {PERF_TYPE_RAW, EVENT_PEND_MISS,       }, "pmiss/elem",     [](double val, double cycles, size_t sum) -> double { return val / sum; } },
  // { {PERF_TYPE_RAW, EVENT_PEND_MISS_CYCLES,}, "pmiss_cyc/elem", [](double val, double cycles, size_t sum) -> double { return val / sum; } },
  { {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK}, "GHz", [](double val, double cycles, size_t sum) -> double { return cycles / val; } }
#endif
};

std::vector<TypeAndConfig> get_wrapper(const std::vector<column_spec>& cols) {
  std::vector<TypeAndConfig> ret;
  for (auto& e : cols) {
    ret.push_back(e.event);
  }
  return ret;
}

LinuxEventsWrapper unified = get_wrapper(all_columns);

namespace fastscancount {
void scancount(const std::vector<const std::vector<uint32_t>*> &data,
               std::vector<uint32_t> &out, size_t threshold) {
  uint64_t largest = 0;
  for(auto z : data) {
    const std::vector<uint32_t> & v = *z;
    if(v[v.size() - 1] > largest) largest = v[v.size() - 1];
  }
  std::vector<uint8_t> counters(largest+1);
  out.clear();
  for (size_t c = 0; c < data.size(); c++) {
    const std::vector<uint32_t> &v = *data[c];
    for (size_t i = 0; i < v.size(); i++) {
      counters[v[i]]++;
    }
  }
  for (uint32_t i = 0; i < counters.size(); i++) {
    if (counters[i] > threshold)
      out.push_back(i);
  }
}
}

using fastscancount::scancount;

void calc_boundaries(uint32_t largest, uint32_t range_size,
                    const std::vector<uint32_t>& data,
                    std::vector<uint32_t>& range_ends) {
  if (!range_size) {
    throw std::runtime_error("range_size must be > 0");
  }
  uint32_t end = 0;
  range_ends.clear();

  for (uint32_t start = 0; start <= largest; start += range_size) {
    uint32_t curr_max = std::min(largest, start + range_size - 1);
    while (end < data.size() && data[end] <= curr_max) {
      end++;
    }
    range_ends.push_back(end);
  }
}

const uint32_t range_size_avx512 = 40000;

void calc_alldata_boundaries(const std::vector<std::vector<uint32_t>>& data,
                             std::vector<std::vector<uint32_t>>& range_ends,
                             size_t range_size) {
  uint32_t largest = get_largest(data);
  range_ends.clear();
  range_ends.resize(data.size());

  for (unsigned i = 0; i < data.size(); ++i) {
    calc_boundaries(largest, range_size, data[i], range_ends[i]);
  }
}

template <typename F>
void test(F f, const std::vector<const std::vector<uint32_t>*>& data_ptrs,
          std::vector<uint32_t>& answer, unsigned threshold, const std::string &name) {
  answer.clear();
  fastscancount::scancount(data_ptrs, answer, threshold);
  size_t s1 = answer.size();
  auto a1 (answer);
  std::sort(a1.begin(), a1.end());
  answer.clear();
  f();
  size_t s2 = answer.size();
  auto a2 (answer);
  std::sort(a2.begin(), a2.end());
  if (a1 != a2) {
    std::cout << "s1: " << s1 << " s2: " << s2 << std::endl;
    auto minsize = std::min(s1, s2);
    if (std::equal(a1.begin(), a1.begin() + minsize, a2.begin())) {
      assert(s1 != s2);
      // one array is a prefix of the other, don't print them all out
      bool refextra = s1 > s2;
      auto &extra = refextra ? a1 : a2;
      for (size_t i = minsize; i < std::max(s1, s2); i++) {
        std::cout << i << " " << extra[i] << '\n';
      }
      throw std::runtime_error(std::string("bug (") + (refextra ? "ref" : "answer") + " had extra elems): " + name);
    } else {
      for(size_t j = 0; j < minsize; j++) {
        std::cout << j << " " << a1[j] << " vs " << a2[j] ;

        if(a1[j] != a2[j]) std::cout << " oh oh ";
        std::cout << std::endl;
      }
      throw std::runtime_error("bug: " + name);
    }
  }
}

template <typename F>
void bench(F f, const std::string &name,
           float& elapsed,
           std::vector<uint32_t> &answer, size_t sum, size_t expected,
           bool print) {
  WallClockTimer tm;
  unified.start();
  f();
  unified.end();
  elapsed += tm.split();
  if (answer.size() != expected)
    std::cerr << "bug: expected " << expected << " but got " << answer.size()
              << "\n";
#ifdef USE_COUNTERS
  if (print) {
    std::ios_base::fmtflags flags(std::cout.flags());
    std::cout << std::setw(NAME_WIDTH) << name;
    std::cout << std::fixed << std::setprecision(4);
    for (auto& col : all_columns) {
      std::cout << std::setw(COL_WIDTH) << col.get_result(unified, sum);
    }
    std::cout.flags(flags);
    std::cout << '\n';
  }
#endif
}

void print_headers() {
  std::cout << std::setw(NAME_WIDTH) << "algorithm";
  for (auto &col : all_columns) {
    std::cout << std::setw(COL_WIDTH)  << col.desc;
  }
  std::cout << '\n';
}

#ifdef RUNNINGTESTS
#define TEST(fn, ...) \
    test(       \
      [&](){    \
        fn(data_ptrs, answer, threshold, ## __VA_ARGS__);     \
      }, data_ptrs, answer, threshold, #fn  \
    );
#else
#define TEST(fn, ...)
#endif

#define BENCH(fn, name, elapsed, ...) \
  answer.clear();       \
  bench(                \
    [&]() {             \
      fn(data_ptrs, answer, threshold, ## __VA_ARGS__ ); \
    },                  \
    name, elapsed, answer, sum, expected, print);

#define BENCHTEST(fn, name, elapsed, ...) TEST(fn, ## __VA_ARGS__) BENCH(fn, name, elapsed, ##__VA_ARGS__)

#define BENCH_LOOP(fn, name, elapsed, ...)  \
  TEST(fn, ## __VA_ARGS__) \
  for (size_t t = 0; t < REPEATS; t++) {    \
    bool print = (t == REPEATS - 1);         \
    BENCH(fn, name, elapsed, ## __VA_ARGS__ ); \
  } \

void demo_data(const std::vector<std::vector<uint32_t>>& data,
               const std::vector<std::vector<uint32_t>>& queries,
               size_t threshold) {

  using namespace fastscancount;

  size_t N = 0;
  for (const auto& data_elem : data) {
    size_t sz = data_elem.size();
    if (sz) {
      N = std::max(N, (size_t)data_elem[sz-1] + 1);
    }
  }

  std::vector<uint32_t> answer;
  answer.reserve(N);

  std::vector<std::vector<uint32_t>> range_boundaries;
  calc_alldata_boundaries(data, range_boundaries, range_size_avx512);

  // aux data for bitscan
  auto bitscan_aux32 = fastscancount::get_all_aux_bitscan<uint32_t>(data);

  auto avx2b_aux32 = fastscancount::implb::get_all_aux<uint32_t>(data);
  auto avx2b_aux16 = fastscancount::implb::get_all_aux<uint16_t>(data);

  std::vector<const std::vector<uint32_t>*> data_ptrs;
  std::vector<const std::vector<uint32_t>*> range_ptrs;

  float elapsed = 0, elapsed_fast = 0, elapsed_avx = 0, elapsed_avx2b32 = 0, elapsed_avx2b16 = 0,
      elapsed_avx2b16b = 0, elapsed_avx2bl16 = 0, elapsed_bitscan = 0, elapsed_avx512 = 0, dummy = 0;

  size_t sum_total = 0;

  size_t qcount = std::min(MAX_QUERIES, queries.size());

  for (size_t qid = 0; qid < qcount; ++qid) {
    const auto& query_elem = queries[qid];
    data_ptrs.clear();
    range_ptrs.clear();
    size_t sum = 0;
    for (uint32_t idx : query_elem) {
      if (idx >= data.size()) {
        std::stringstream err;
        err << "Inconsistent data, posting " << idx <<
               " is >= # of postings " << data.size() << " query id " << qid;
        throw std::runtime_error(err.str());
      }
      sum += data[idx].size();
      data_ptrs.push_back(&data[idx]);
      range_ptrs.push_back(&range_boundaries[idx]);
    }
    sum_total += sum;

    fastscancount::fastscancount(data_ptrs, answer, threshold);
    const size_t expected = answer.size();

    std::cout << "Qid: " << qid << " got " << expected << " hits [thresh = "
        << threshold << ", array count = " << data_ptrs.size() << "]\n";

    bool print = PRINT_ALL || (qid == qcount - 1);

    if (print)
      print_headers();

    // BENCHTEST(scancount, "baseline scancount", elapsed);
    BENCHTEST(fastscancount::fastscancount, "cache-sensitive scancount", elapsed_fast);

#ifdef __AVX2__
    BENCHTEST(fastscancount_avx2, "AVX2-based scancount", elapsed_avx);

    BENCHTEST(fastscancount_avx2b32<fastscancount::record_hits_c>, "Try2 AVX2 in C", dummy, avx2b_aux32, query_elem);

    BENCHTEST((fastscancount_avx2b<uint32_t, fastscancount::record_hits_asm_branchy32>), "AVX2B ASM branchy    32b", elapsed_avx2b32,  avx2b_aux32, query_elem);
    BENCHTEST((fastscancount_avx2b<uint16_t, fastscancount::record_hits_asm_branchy16>), "AVX2B ASM branchy    16b", elapsed_avx2b16, avx2b_aux16, query_elem);
    // BENCHTEST((fastscancount_avx2b<uint16_t, fastscancount::record_hits_asm_branchyB >), "AVX2B ASM branchy      B", elapsed_avx2b16b, avx2b_aux16, query_elem);

    // BENCHTEST((fastscancount_avx2b<uint32_t, fastscancount::record_hits_asm_branchless32>), "AVX2B ASM branchless 32b", dummy, avx2b_aux32, query_elem);
    // BENCHTEST((fastscancount_avx2b<uint16_t, fastscancount::record_hits_asm_branchless16>), "AVX2B ASM branchless 16b", elapsed_avx2bl16, avx2b_aux16, query_elem);
#endif
#ifdef __AVX512F__
    BENCHTEST(bitscan_avx512, "bitscan_avx512", elapsed_bitscan, bitscan_aux32, query_elem);
    BENCHTEST(fastscancount_avx512, "AVX512-based scancount", elapsed_avx512, range_size_avx512, range_ptrs);
#endif
  }

  std::cout << std::fixed;
#define ELAPSEDOUT(var) std::setprecision(0) << (sum_total/(var/1e3)) \
    << std::setprecision(2) << std::setw(8) << (elapsed_fast/var) << '\n'
  std::cout << "Algorithm        Elems/ms    Speedup vs fastscancount" << std::endl;
  std::cout << "scancount:        " << std::setw(8) << ELAPSEDOUT(elapsed);
  std::cout << "fastscancount:    " << std::setw(8) << ELAPSEDOUT(elapsed_fast);
#ifdef __AVX2__
  std::cout << "fastscan_avx2    :" << std::setw(8) << ELAPSEDOUT(elapsed_avx);
  std::cout << "fastscan_avx2bb  :" << std::setw(8) << ELAPSEDOUT(elapsed_avx2b32);
  std::cout << "fastscan_avx2b16 :" << std::setw(8) << ELAPSEDOUT(elapsed_avx2b16);
  std::cout << "fastscan_avx2b16B:" << std::setw(8) << ELAPSEDOUT(elapsed_avx2b16b);
  std::cout << "fastscan_avx2bl16:" << std::setw(8) << ELAPSEDOUT(elapsed_avx2bl16);
#endif
#ifdef __AVX512F__
  std::cout << "bitscan_avx512:   " << std::setw(8) << ELAPSEDOUT(elapsed_bitscan);
  std::cout << "fastsct_avx512:   " << std::setw(8) << ELAPSEDOUT(elapsed_avx512);
#endif

  std::cout << std::flush;
}

#define OUT(x) std::cout << #x ": " << x << '\n';

void demo_random(size_t N, size_t length, size_t array_count, size_t threshold) {

  using namespace fastscancount;

  std::vector<std::vector<uint32_t>> data(array_count);

  std::vector<const std::vector<uint32_t>*> data_ptrs;
  std::vector<uint32_t> answer;
  answer.reserve(N);

  size_t sum = 0;
  for (size_t c = 0; c < array_count; c++) {
    std::vector<uint32_t> &v = data[c];
    // uncomment this (and comment out the following loop)
    // for more random lenths and values
    // size_t len = rand() % length;
    // // size_t len = length;
    // size_t n = rand() % N;
    // for (size_t i = 0; i < len; i++) {
    //   v.push_back(rand() % n);
    // }

    for (size_t i = 0; i < length; i++) {
      v.push_back(rand() % N);
    }

    std::sort(v.begin(), v.end());
    v.resize(std::distance(v.begin(), unique(v.begin(), v.end())));
    // // make each vector a multiple of unroll, simplying other logic
    // while (v.size() % fastscancount::unroll != 0) {
    //   v.pop_back();
    // }
    v.shrink_to_fit();
    sum += v.size();
    data_ptrs.push_back(&data[c]);
  }

  OUT(sum);

  // aux data for AVX-512
  std::vector<std::vector<uint32_t>> range_boundaries;
  calc_alldata_boundaries(data, range_boundaries, range_size_avx512);
  std::vector<const std::vector<uint32_t>*> range_ptrs;
  for (size_t c = 0; c < array_count; c++) {
    range_ptrs.push_back(&range_boundaries[c]);
  }

  // aux data for avx2b
  // auto avx2b_aux32 = fastscancount::implb::get_all_aux<uint32_t>(data);
  // auto avx2b_aux16 = fastscancount::implb::get_all_aux<uint16_t>(data);

  // aux data for bitscan
  auto bitscan_aux32 = fastscancount::get_all_aux_bitscan<uint32_t>(data);

  // query definition composed of all the arrays
  std::vector<uint32_t> query_elem(data.size());
  std::iota(query_elem.begin(), query_elem.end(), 0);

  float elapsed = 0, elapsed_fast = 0, elapsed_avx = 0,
      elapsed_avx2bb = 0, elapsed_avx2b16 = 0, elapsed_avx512 = 0, dummy = 0;
  fastscancount::scancount(data_ptrs, answer, threshold);
  const size_t expected = answer.size();
  std::cout << "Got " << expected << " hits\n";
  size_t sum_total = sum * REPEATS;

  print_headers();

  // BENCH_LOOP(scancount, "baseline scancount", elapsed);
  BENCH_LOOP(fastscancount::fastscancount, "fastscancount", elapsed_fast);

  // BENCH_LOOP(bitscan_scalar, "bitscan_scalar", dummy, bitscan_aux32, query_elem);
  BENCH_LOOP(bitscan_fake,  "bitscan_fake", dummy, bitscan_aux32, query_elem);
#ifdef __AVX512F__
  BENCH_LOOP(bitscan_avx512,  "bitscan_avx512", dummy, bitscan_aux32, query_elem);
#endif

#ifdef __AVX2__
  BENCH_LOOP(fastscancount_avx2,  "AVX2-based scancount", elapsed_avx);

  // BENCH_LOOP((fastscancount_avx2b<uint32_t, fastscancount::record_hits_c>), "AVX2B in C 32b", dummy, avx2b_aux32, query_elem);
  // BENCH_LOOP((fastscancount_avx2b<uint16_t, fastscancount::record_hits_c>), "AVX2B in C 16b", dummy, avx2b_aux16, query_elem);

  // BENCH_LOOP((fastscancount_avx2b<uint32_t, fastscancount::record_hits_asm_branchy32>), "AVX2B ASM branchy    32b", elapsed_avx2bb, avx2b_aux32, query_elem);
  // BENCH_LOOP((fastscancount_avx2b<uint16_t, fastscancount::record_hits_asm_branchy16>), "AVX2B ASM branchy    16b", elapsed_avx2b16, avx2b_aux16, query_elem);
  // BENCH_LOOP((fastscancount_avx2b<uint16_t, fastscancount::record_hits_asm_branchyB >), "AVX2B ASM branchy      B", dummy, avx2b_aux16, query_elem);

  // BENCH_LOOP((fastscancount_avx2b<uint32_t, fastscancount::record_hits_asm_branchless32>), "AVX2B ASM branchless 32b", dummy, avx2b_aux32, query_elem);
  // BENCH_LOOP((fastscancount_avx2b<uint16_t, fastscancount::record_hits_asm_branchless16>), "AVX2B ASM branchless 16b", dummy, avx2b_aux16, query_elem);
#endif

#ifdef __AVX512F__
  BENCH_LOOP(fastscancount_avx512, "AVX512-based scancount", elapsed_avx512, range_size_avx512, range_ptrs);
#endif

  std::cout << std::fixed;
#define ELAPSEDOUT(var) std::setprecision(0) << (sum_total/(var/1e3)) \
    << std::setprecision(2) << std::setw(8) << (elapsed_fast/var) << '\n'
  std::cout << "Algorithm        Elems/ms   Speedup vs fastscancount" << std::endl;
  std::cout << "scancount:       " << std::setw(8) << ELAPSEDOUT(elapsed);
  std::cout << "fastscancount:   " << std::setw(8) << ELAPSEDOUT(elapsed_fast);
#ifdef __AVX2__
  std::cout << "fastscan_avx2:   " << std::setw(8) << ELAPSEDOUT(elapsed_avx);
  std::cout << "fastscan_avx2bb: " << std::setw(8) << ELAPSEDOUT(elapsed_avx2bb);
  std::cout << "fastscan_avx2b16:" << std::setw(8) << ELAPSEDOUT(elapsed_avx2b16);
#endif
#ifdef __AVX512F__
  std::cout << "fastsct_avx512:  " << std::setw(8) << ELAPSEDOUT(elapsed_avx512);
#endif

  std::cout << std::flush;
}

void usage(const std::string& err="") {
  if (!err.empty()) {
    std::cerr << err << std::endl;
  }
  std::cerr << "usage: --postings <postings file> --queries <queries file> --threshold <threshold>" << std::endl;
}

int main(int argc, char *argv[]) {
  // A very naive way to process arguments,
  // but it's ok unless we need to extend it substantially.
  if (argc != 1) {
    if (argc != 7) {
      usage("");
      return EXIT_FAILURE;
    }
    std::string postings_file, queries_file;
    int threshold = -1;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--postings") {
        postings_file = argv[++i];
      } else if (std::string(argv[i]) == "--queries") {
        queries_file = argv[++i];
      } else if (std::string(argv[i]) == "--threshold") {
        threshold = std::atoi(argv[++i]);
      }
    }
    if (postings_file.empty() || queries_file.empty() || threshold < 0) {
      usage("Specify queries, postings, and the threshold!");
      return EXIT_FAILURE;
    }
    std::vector<uint32_t> tmp;
    std::vector<std::vector<uint32_t>> data;
    {
      MaropuGapReader drdr(postings_file);
      if (!drdr.open()) {
        usage("Cannot open: " + postings_file);
        return EXIT_FAILURE;
      }
      while (drdr.loadIntegers(tmp)) {
        data.push_back(tmp);
      }
    }

    std::vector<std::vector<uint32_t>> queries;
    {
      MaropuGapReader qrdr(queries_file);
      if (!qrdr.open()) {
        usage("Cannot open: " + queries_file);
        return EXIT_FAILURE;
      }
      while (qrdr.loadIntegers(tmp)) {
        queries.push_back(tmp);
      }
    }

    std::cout << "Read " << data.size() << " posting arrays and " << queries.size() << " queries\n";

    if (do_analyze) {
      analyze(data, queries);
    }

    try {
      demo_data(data, queries, threshold);
    } catch (const std::exception& e) {
      std::cerr << "Exception: " << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    try {
      // Previous demo with threshold 3
      //demo_random(20000000, 50000, 100, 3);
      for (unsigned k = START_THRESHOLD; k < END_THRESHOLD; ++k) {
        std::cout << "RAND_MAX is " << RAND_MAX << std::endl;
        std::cout << "Demo threshold:" << k << std::endl;
        demo_random(DEMO_DOMAIN, DEMO_ARRAY_SIZE, DEMO_ARRAY_COUNT, k);
        std::cout << "=======================" << std::endl;
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception: " << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
