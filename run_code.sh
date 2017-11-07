#!/usr/bin/env bash 

set -e

r=0
mp=0
rf=''
mpf=''
rs=''
mps=''
b=1
clip=''
clips=''
gpu=''

glove_index=1
gloves=(6B.100d 6B.300d 840B.300d 6B.50d)


if [ "$#" -lt 4 ]; then
	echo "Usage: $0 <Pattern spcification> <MLP dim> <Learning rate> <dropout> <reschedule=$r> <maxplus=$mp> <batch size=$b> <gradient clipping (optional)> <gpu (optional)> <glove index=$glove_index (${gloves[@]})>"
	exit -1
elif [ "$#" -gt 4 ]; then
	r=$5

	if [ $r -eq 1 ]; then
		rf="-r"
		rs='_r'
	fi
	if [ "$#" -gt 5 ]; then
		mp=$6
		if [ $mp -eq 1 ]; then
			mpf="--maxplus"
			mps='_mp'
		fi
		if [ "$#" -gt 6 ]; then
			b=$7
			if [ "$#" -gt 7 ]; then
				clip="--clip $8"
				clips="_clip$8"
				if [ "$#" -gt 8 ]; then
					if [ $9 -eq 1 ]; then
						gpu='-g'
					fi
					if [ "$#" -gt 9 ]; then
						glove_index=${10}
					fi
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

glove=${gloves[$glove_index]}

git_tag=$(git log | head -1 | awk '{print $2}' | cut -b-7)

odir=$HOME/work/soft_patterns/output_p${p2}_d${dim}_l${lr}_t${t}${rs}${mps}_b${7}${clips}_${glove}_$git_tag


mkdir -p $odir

com="python -u soft_patterns.py        \
         -e $HOME/resources/glove/glove.${glove}.txt         \
        --td $HOME/resources/text_cat/stanford_sentiment_binary//train.data         \
        --tl $HOME/resources/text_cat/stanford_sentiment_binary//train.labels       \
        --vd $HOME/resources/text_cat/stanford_sentiment_binary//dev.data           \
        --vl $HOME/resources/text_cat/stanford_sentiment_binary//dev.labels         \
        --model_save_dir $odir \
        -i 250 \
         -p $p \
        -t $t \
        -d $dim \
        -l $lr $rf $mpf $clip $gpu\
        -b $b"

echo $com
$com | tee $odir/output.dat
