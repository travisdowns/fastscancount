#!/bin/bash

set -e

mkdir -p ./tmp

DATADIR=${DATADIR:-../data}

DIRS=${DIRS:-$(cd $DATADIR; echo $DATADIR/*)}
echo "Using DATADIR=$DATADIR DIRS=$DIRS"

export CXXEXTRA=-DNO_COUNTERS

make clean
make -j

for dir in $DIRS; do
    OUTFILE=tmp/result-$(basename -- "$dir").out
    rm -f src/asm-kernels.o
    echo "DIR=$dir OUTFILE=$OUTFILE"
    (set -x; ./counter --postings $dir/postings.bin --queries $dir/queries.bin --threshold 5 > $OUTFILE)
    tail $OUTFILE
done
