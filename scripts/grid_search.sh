#!/usr/bin/env  bash

if [ $# -lt 2 ]; then
        echo "Usage: $0 <dataset> <seed>"
fi

set -e

for p in 6:20,5:20,4:10,3:10,2:10 6:10,5:10,4:10,3:10,2:10 7:10,6:10,5:10,4:10,3:10,2:10; do
        for l in 0.01 0.005 0.001; do
                for t in 0.05 0.1 0.2; do
                        for d in 10 25 50 100; do
                                export CUDA_VISIBLE_DEVICES=2 && ./run_code.sh $p $d $l $t 1 2 150  0 1 2 0 0.1 $1 $2
                        done
                done
        done
done
