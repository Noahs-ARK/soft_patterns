#!/usr/bin/env bash 

set -e

r=0
mp=0
rf=''
mpf=''
rs=''
mps=''
b=1
gpu=''

if [ "$#" -lt 4 ]; then
	echo "Usage: $0 <Pattern spcification> <MLP dim> <Learning rate> <dropout> <reschedule=$r> <maxplus=$mp> <batch size=$b> <gpu (optional)>"
	exit -1
elif [ "$#" -gt 4 ]; then
	r=$5

	if [ $r -eq 1 ]; then
		rf="-r"
		rs='_r'
	fi
	if [ "$#" -gt 5 ]; then
		mp=$5
		if [ $mp -eq 1 ]; then
			mpf="--maxplus"
			mps='_mp'
		fi
		if [ "$#" -gt 6 ]; then
			b=$6
			if [ "$#" -gt 7 ]; then
				if [ $mp -eq 1 ]; then
					gpu='-g'
				fi
			fi
		fi
	fi
fi

p=$1

p2=`echo $p | tr ',' '_'`
dim=$2
lr=$3
t=$4

git_tag=`git log | head -1 | cut -d ' ' -f2`

suffix=p${p2}_d${dim}_l${lr}_t${t}${rs}${mps}_b${b}_$git_tag
odir=~/work/soft_patterns/output_$suffix

mkdir -p $odir

python -u soft_patterns.py        \
	 -e $HOME/resources/glove/glove.6B.100d.txt         \
	--td $HOME/resources/text_cat/stanford_sentiment_binary//train.data         \
	--tl $HOME/resources/text_cat/stanford_sentiment_binary//train.labels       \
	--vd $HOME/resources/text_cat/stanford_sentiment_binary//dev.data           \
	--vl $HOME/resources/text_cat/stanford_sentiment_binary//dev.labels         \
	--model_save_dir $odir \
	-i 250 \
	 -p $p \
	-t $t \
	-d $dim \
	-l $lr $rf $mpf $gpu\
	-b $b |& tee $odir/output.dat
