#!/bin/bash

set -e

mkdir -p ./tmp

DIRS=${DIRS:-$(cd ../data; echo *)}
echo "Using DIRS=$DIRS"

export CXXEXTRA=-DNO_COUNTERS

make clean
make -j

for dir in $DIRS; do
    OUTFILE=tmp/result-$dir.csv
    rm -f src/asm-kernels.o
    echo "DIR=$dir"
    ./counter --csv --postings ../data/$dir/postings.bin --queries ../data/$dir/queries.bin --threshold-range 3 15 > $OUTFILE
done
