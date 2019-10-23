-include local.mk

OPT ?= -O3
NASM ?= nasm
# set this to empty to enable asserts
NDEBUG ?= -DNDEBUG

# Leo really doubts -mavx2 helps anything, but one can
# disable avx512 tests by enforcing -mavx2
#CXXFLAGS := -std=c++17 $(OPT) -mavx2
CXXFLAGS := -std=c++17 $(OPT) -march=native -g $(NDEBUG)

LOCAL_MK = $(wildcard local.mk)

counter: counters.o asm-kernels.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

counters.o: benchmark/counters.cpp include/*.h Makefile $(LOCAL_MK)
	$(CXX) $(CXXFLAGS) $(CXXEXTRA) -c benchmark/counters.cpp -Ibenchmark -Iinclude

asm-kernels.o: asm-kernels.asm $(LOCAL_MK)
	$(NASM) $(NASMFLAGS) -felf64 $<

clean:
	rm -f counter *.o

