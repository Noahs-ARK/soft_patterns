#!/usr/bin/env bash

model=0

if [ $# -lt 1 ]; then
	echo "Usage: $0 <dataset> <model=$model>"
	exit -1
elif [ $# -gt 1 ]; then
	model=$2
fi

dataset=$1
datasetb=$(echo $dataset | sed -rs 's/_[0-9]*$//')

wdir=$HOME/work/soft_patterns/ 

if [ $model -eq 1 ]; then
	wdir=$wdir/baselines/dan_b150/$dataset/
	s='/output.dat'
elif [ $model -eq 2 ]; then
	wdir=$wdir/logs/
	s="_${1}.out"
else
	s="_${1}_seed*/output.dat"
fi

echo "Best dev:"
grep loss: $wdir/*${s} | sort -rnk 18 | head -n1 | awk '{print $18}' 
f=$(grep loss: $wdir/*${s} | sort -nk 16 | head -n 1 | cut -d: -f1)
n=$(grep loss: $wdir/*${s} | sort -nk 16 | head -n 1 | awk '{print $2}')

echo "Test: $f, $n"

./scripts/run_test.sh $f $model $n ~/resources/text_cat/$datasetb/test.{data,labels}
