#!/bin/bash

set -e

mkdir -p ./tmp

DIRS=${DIRS:-$(cd ../data; echo *)}
echo "Using DIRS=$DIRS"

export CXXEXTRA=-DNO_COUNTERS

make clean
make -j

for dir in $DIRS; do
    OUTFILE=tmp/result-$dir.out
    rm -f src/asm-kernels.o
    echo "DIR=$dir"
    ./counter --postings ../data/$dir/postings.bin --queries ../data/$dir/queries.bin --threshold 5 > $OUTFILE
    tail $OUTFILE
done
