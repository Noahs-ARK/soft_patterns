#!/usr/bin/env bash

#set -e

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
m=$HOME/work/soft_patterns/baselines/dan_b150/$1/

ls=(0.1 0.05 0.01 0.005 0.001)
ws=(0.1 0.2 0.3 0.4)
ds=(10 50 100 300)

n=$(echo "${#ls[@]} * ${#ws[@]} * ${#ds[@]}" | bc -l)

echo $n
ind=($(seq 0 $n | sort -R))

ind2=($(seq 0 $n))

for i in {0..19}; do
	ind2[${ind[$i]}]=1
done

for i in $(seq 20 $n); do
	ind2[${ind[$i]}]=0
done

i=-1
for lr in ${ls[@]}; do
	for w in ${ws[@]}; do
		for d in ${ds[@]}; do
		    let i++

            if [ ${ind2[$i]} -eq 0 ]; then
			    echo $i randomed out
			    continue
			fi

			local_d=$m/l${lr}_t${t}_d${d}
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
				-b 150 \
				-d $d \
				-g -r \
				-e $e"
			echo $com
			export CUDA_VISIBLE_DEVICES=1 && $com | tee $local_d/output.dat
		done
	done
done

exit 0
