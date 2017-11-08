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

glove_index=0
gloves=(6B.100d 6B.300d 840B.300d 6B.50d)



if [ -z ${WORK+x} ]; then
    WORK=$HOME/resources/
    model_dir=$HOME/work/soft_patterns/
else
    model_dir=$WORK/
fi

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

s=p${p2}_d${dim}_l${lr}_t${t}${rs}${mps}_b${7}${clips}_${glove}_$git_tag
odir=$model_dir/output_$s


mkdir -p $odir

com="python -u soft_patterns.py        \
         -e $WORK/glove/glove.${glove}.txt         \
        --td $WORK/text_cat/stanford_sentiment_binary//train.data         \
        --tl $WORK/text_cat/stanford_sentiment_binary//train.labels       \
        --vd $WORK/text_cat/stanford_sentiment_binary//dev.data           \
        --vl $WORK/text_cat/stanford_sentiment_binary//dev.labels         \
        --model_save_dir $odir \
        -i 250 \
         -p $p \
        -t $t \
        -d $dim \
        -l $lr $rf $mpf $clip $gpu\
        -b $b"

echo $com

function gen_cluster_file {
    local s=$1

    f=$HOME/work/soft_patterns/runs/$s

    echo "SBATCH -J $s" > $f
    echo "SBATCH -o $s.%j.out" >> $f
    echo "SBATCH -p normal" >> $f         # specify queue
    echo "SBATCH -N 1" >> $f              # Number of nodes, not cores (16 cores/node)
    echo SBATCH -t 24:00:00 >> $f       # max time

    echo "SBATCH --mail-user=roysch@cs.washington.edu" >> $f
    echo "SBATCH --mail-type=ALL" >> $f

    echo "SBATCH -A TG-DBS110003       # project/allocation number;" >> $f
    echo "source activate torch3" >> $f

    echo "mpirun $com" >> $f

    echo $f
}

if [[ "$HOSTNAME" == *.stampede2.tacc.utexas.edu ]]; then
    f=$(gen_cluster_file $s)

    sbatch $f
else
    $com | tee $odir/output.dat
fi
