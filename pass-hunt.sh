#!/bin/bash

set -e

mkdir -p ./tmp

START=${START:-128}
INC=${INC:-128}
END=${END:-2048}
echo "Using START=$START INC=$INC END=$END"

DIRS=${DIRS:-$(cd ../data; echo *)}
echo "Using DIRS=$DIRS"

rm -f tmp/build.log

# export CXXEXTRA=-DNO_COUNTERS

make clean
make -j "CXXEXTRA=-DNO_COUNTERS"

for dir in $DIRS; do
    for pass_size in $(seq $START $INC $END); do
        OUTFILE=tmp/pass-$pass_size.out
        echo "DIR=$dir CHUNKS_PER_PASS=$pass_size"
        rm -f src/bitscan.o
        make "CXXEXTRA=-DCHUNKS_PER_PASS=$pass_size -DNO_COUNTERS" >> tmp/build.log
        ./counter --postings ../data/$dir/postings.bin --queries ../data/$dir/queries.bin --threshold 5 > $OUTFILE
        grep '_asm:' $OUTFILE
    done
done
