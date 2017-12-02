#!/usr/bin/env bash

set -e

if [ $# -lt 1 ]; then
	echo "Usage: $0 <dataset (amazon_reviews, stanford_sentiment_binary, ROC_stories)>"
	exit -1
fi

base_dir=$HOME/resources/
data_dir=$base_dir/text_cat/$1/

td=$data_dir/train.data
vd=$data_dir/dev.data
tl=$data_dir/train.labels
vl=$data_dir/dev.labels

e=$base_dir/glove/glove.840B.300d.txt
m=$HOME/work/soft_patterns/baselines/lstm/$1/


for lr in 0.01 0.005 0.001; do
	for t in 0.05 0.1 0.2; do
		for d in 10 25 50 100; do
		    for h in 100 200 300; do
                local_d=$m/l${lr}_t${t}_d${d}_h${h}
                mkdir -p  $local_d
                com="python -u baselines/lstm.py \
                    --td $td \
                    --vd $vd \
                    --tl $tl \
                    --vl $vl \
                    -m $local_d 	\
                    -l $lr \
                    -t $t \
                    --hidden_dim $h \
                    -i 250 \
                    -b 150 \
                    -d $d \
                    -g -r \
                    -e $e"
                echo $com
                $com | tee $local_d/output.dat
#                break
            done
		done
	done
done

exit 0
