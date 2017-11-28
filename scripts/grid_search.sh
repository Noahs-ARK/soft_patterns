#!/usr/bin/env  bash

if [ $# -lt 2 ]; then
	echo "Usage: $0 <dataset> <seed>"
	exit -1
fi

set -e

dirs=(stanford_sentiment_binary amazon_reviews ROC_stories)
dir=${dirs[$1]}

for p in 6:20,5:20,4:10,3:10,2:10 6:10,5:10,4:10,3:10,2:10 7:10,6:10,5:10,4:10,3:10,2:10; do
        for l in 0.01 0.005 0.001; do
                for t in 0.05 0.1 0.2; do
                        for d in 10 25 50 100; do
				p2=$(echo ${p} | tr ',' '_' | tr ':' '-')
                                s=$HOME/work/soft_patterns/output_p${p2}_d${d}_l${l}_t${t}_r_mt_b150_clip0_840B.300d_w0.1_${dir}_seed${2}

#				ls ${s}*/output.dat
				v=$(ls ${s}*/output.dat |& grep 'No such file' | wc -l)

				if [ $v -gt 0 ]; then
#					echo run
#					exit -1
					export CUDA_VISIBLE_DEVICES=2 && ./scripts/run_code.sh $p $d $l $t 1 2 150  0 1 2 0 0.1 $1 $2
				fi
                        done
                done
        done
done
