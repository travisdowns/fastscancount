#!/bin/bash

set -e

mkdir -p ./tmp

DIRS=${DIRS:-$(cd ../data; echo *)}
echo "Using DIRS=$DIRS"

rm -f tmp/build.log

export CXXEXTRA=-DNO_COUNTERS

make clean
make -j

for dir in $DIRS; do
    for instr in prefetcht0; do
        for pf in $(seq 0 64 512); do
            rm -f src/asm-kernels.o
            make "NASMEXTRA=-Dpf_offset=$pf -Dpf_instr=$instr" >> tmp/build.log 2>&1
            echo "DIR=$dir Prefetch distance=$pf "
            ./counter --postings ../data/$dir/postings.bin --queries ../data/$dir/queries.bin --threshold 5 > tmp/pf-$pf.out
            grep '_asm:' tmp/pf-$pf.out
        done
    done
done
