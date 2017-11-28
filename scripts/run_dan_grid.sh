#!/usr/bin/env bash

set -e

if [ $# -lt 1 ]; then
	echo "Usage: $0 <dataset>"
	exit -1
fi

base_dir=$HOME/resources/
data_dir=$base_dir/text_cat/$1/

td=$data_dir/train.data
vd=$data_dir/dev.data
tl=$data_dir/train.labels
vl=$data_dir/dev.labels

e=$base_dir/glove/glove.840B.300d.txt
m=$HOME/work/soft_patterns/baselines/dan/$1/


for lr in 0.1 0.05 0.01 0.005 0.001; do
	for w in 0.1 0.2 0.3 0.4; do
		for d in 10 50 100 300; do
			local_d=$m/l${lr}_w${w}_d${d}
			mkdir -p  $local_d
			com="python -u baselines/dan.py \
				--td $td \
				--vd $vd \
				--tl $tl \
				--vl $vl \
				-m $local_d 	\
				-l $lr \
				-w $w \
				-i 250 \
				-d $d \
				-g \
				-e $e"
			echo $com
			$com | tee $local_d/output.dat
		done
	done
done

exit 0
