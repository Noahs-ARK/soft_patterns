#!/usr/bin/env bash

model=0

# dev loss index
ind=16
if [ $# -lt 1 ]; then
	echo "Usage: $0 <dataset> <model=$model> <maximize accuracy rather than loss>"
	exit -1
elif [ $# -gt 1 ]; then
	model=$2
	if [ $# -gt 2 ]; then
		# dev acc index
		ind='18 -r'
	fi
fi

dataset=$1
datasetb=$(echo $dataset | sed -rs 's/_[0-9]*$//')

wdir=$HOME/work/soft_patterns/ 

if [ $model -eq 1 ]; then
	wdir=$wdir/baselines/dan_b150/$dataset/
	s='/output.dat'
elif [ $model -eq 2 ]; then
	wdir=$wdir/logs/
	s="_${1}[\._]*out"
else
	s="_${1}_seed*/output.dat"
fi

l=$(ls $wdir/*${s} | wc -l | awk '{print $1}')
echo "$l files found. Latest:"

ls -ltr $wdir/*${s} | tail -n1 | cut -d' ' -f6-9

echo "Best dev:"

grep loss: $wdir/*${s} | sort -rnk 18 | head -n1 | awk '{print $18}' 
f=$(grep loss: $wdir/*${s} | sort -nk $ind | head -n 1 | cut -d: -f1)
n=$(grep loss: $wdir/*${s} | sort -nk $ind | head -n 1 | awk '{print $2}')

echo "Test: $f, $n"

./scripts/run_test.sh $f $model $n ~/resources/text_cat/$datasetb/test.{data,labels}
