-include local.mk

# disable built-in rules
.SUFFIXES:

OPT ?= -O3
NASM ?= nasm
# set this to empty to enable asserts
NDEBUG ?= -DNDEBUG
MARCH ?= native

# Leo really doubts -mavx2 helps anything, but one can
# disable avx512 tests by enforcing -mavx2
#CXXFLAGS := -std=c++17 $(OPT) -mavx2
CXXFLAGS := -std=c++17 $(OPT) -march=$(MARCH) -g $(NDEBUG) -DENABLE_TIMER -MMD -Wno-ignored-attributes

SRC := $(wildcard src/*.cpp src/*.asm)
OBJ := $(SRC:.cpp=.o)
OBJ := $(OBJ:.asm=.o)

BENCH_SRC := $(wildcard benchmark/*.cpp)
BENCH_OBJ := $(BENCH_SRC:.cpp=.o)

TEST_SRC := $(wildcard test/*.cpp)
TEST_OBJ := $(TEST_SRC:.cpp=.o)

DEPS := $(patsubst %.o,%.d,$(OBJ) $(BENCH_OBJ) $(TEST_OBJ))

MAKE_DEPS := Makefile $(wildcard local.mk)

CXX_RULE = $(CXX) $(CXXFLAGS) $(CXXEXTRA) -c $< -o $@ -Iinclude -Iinclude/boost

# $(info SRC=$(SRC))
$(info OBJ=$(OBJ))
# $(info TEST_OBJ=$(TEST_OBJ))
# $(info DEPS=$(DEPS))

all: counter unit-test

-include $(DEPS)

unit-test: $(OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

counter: $(OBJ) $(BENCH_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

benchmark/%.o : benchmark/%.cpp benchmark/*.h* include/*.h* $(MAKE_DEPS)
	$(CXX_RULE) -Ibenchmark

src/%.o : src/%.cpp $(MAKE_DEPS)
	$(CXX_RULE)

test/%.o : test/%.cpp $(MAKE_DEPS)
	$(CXX_RULE)

src/%.o: src/%.asm $(MAKE_DEPS)
	$(NASM) $(NASMFLAGS) $(NASMEXTRA) -felf64 $<

clean:
	rm -f counter unit-test src/*.[od] benchmark/*.[od] test/*.[od]

