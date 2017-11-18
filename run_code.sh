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
file_type=0
self_loop_scale=0
epsilon_scale=0
w=0
mtf=''
mts=''
glove_index=0
gloves=(6B.100d 6B.300d 840B.300d 6B.50d)



if [ -z ${WORK+x} ]; then
    WORK=$HOME/
    if [ -z ${model_dir+x} ]; then
        model_dir=$HOME/work/soft_patterns
    fi
else
    if [ -z ${model_dir+x} ]; then
        model_dir=${WORK}/soft_patterns
    fi
fi

if [ -z ${data_dir+x} ]; then
    data_dir=${WORK}/resources
fi

sst_dir="${data_dir}/text_cat/stanford_sentiment_binary"
glove_dir="${data_dir}/glove"



suffix=''

if [ "$#" -lt 4 ]; then
	echo "Usage: $0 <Pattern specification> <MLP dim> <Learning rate> <dropout> <reschedule=$r> <maxplus=$mp (1 for maxplus, 2 for maxtimes, 0 for prob)> <batch size=$b> <gradient clipping (optional)> <gpu (optional)> <glove index=$glove_index (${gloves[@]})> <file type=$file_type (0 -- lower case, 1 -- case sensitive, 2 -- train with phrases, 3 -- fine grained categories, 4 -- fine grained categories with phrases)> <self loop scale=$self_loop_scale> <epsilon scale=$epsilon_scale> <word_dropout=$w>"
	exit -1
elif [ "$#" -gt 4 ]; then
	r=$5

	if [ ${r} -eq 1 ]; then
		rf="-r"
		rs='_r'
	fi
	if [ "$#" -gt 5 ]; then
		mp=$6
		if [ ${mp} -eq 1 ]; then
			mpf="--maxplus"
			mps='_mp'
		elif [ ${mp} -eq 2 ]; then
			mpf="--maxtimes"
			mps='_mt'
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
						if [ "$#" -gt 10 ]; then
							if [ "${11}" -eq 1 ]; then
								suffix='_case_sensitive'
							elif [ "${11}" -eq 2 ]; then
								suffix='_phrases'
							elif [ "${11}" -eq 3 ]; then
								suffix='_fine'
							elif [ "${11}" -eq 4 ]; then
								suffix='_phrases_fine'
							elif [ "${11}" -ne 0 ]; then
								echo "Expected a number between 0-4 for file type, got ${11}"
								exit -2
							fi
							if [ "$#" -gt 11 ]; then
							    self_loop_scale=${12}
     				                           if [ "$#" -gt 12 ]; then
                                				epsilon_scale=${13}
     				                           	if [ "$#" -gt 13 ]; then
                                				    w=${14}
                                			   	fi
                                			   fi
                             				fi
						fi
					fi
				fi
			fi
		fi
	fi
fi

p=$1

p2=`echo ${p} | tr ',' '_' | tr ':' '-'`
dim=$2
lr=$3
t=$4

glove=${gloves[$glove_index]}

git_tag=$(git log | head -1 | awk '{print $2}' | cut -b-7)

s=p${p2}_d${dim}_l${lr}_t${t}${rs}${mps}_b${7}${clips}_${glove}${suffix}_w${w}_slScale${self_loop_scale}_epsScale${epsilon_scale}_${git_tag}
odir=${model_dir}/output_${s}


mkdir -p ${odir}

com="python -u soft_patterns.py        \
         -e ${glove_dir}/glove.${glove}.txt         \
        --td ${sst_dir}/train$suffix.data    \
        --tl ${sst_dir}/train$suffix.labels  \
        --vd ${sst_dir}/dev$suffix.data      \
        --vl ${sst_dir}/dev.labels           \
        --model_save_dir $odir \
        -i 250 \
         -p $p \
        -t $t \
        -d $dim \
        -l $lr $rf $mpf $clip $gpu\
        --epsilon_scale_value $epsilon_scale \
        --self_loop_scale_value $self_loop_scale \
        -b $b \
	-w $w"

echo ${com}

function gen_cluster_file {
    local s=$1

    f=$HOME/work/soft_patterns/runs/${s}

    echo "#!/usr/bin/env bash" > ${f}
    echo "#SBATCH -J $s" >> ${f}
    echo "#SBATCH -o $HOME/work/soft_patterns/logs/$s.out" >> ${f}
    echo "#SBATCH -p normal" >> ${f}         # specify queue
    echo "#SBATCH -N 1" >> ${f}              # Number of nodes, not cores (16 cores/node)
    echo "#SBATCH -n 1" >> ${f}
    echo "#SBATCH -t 24:00:00" >> ${f}       # max time

    echo "#SBATCH --mail-user=roysch@cs.washington.edu" >> ${f}
    echo "#SBATCH --mail-type=ALL" >> ${f}

    echo "#SBATCH -A TG-DBS110003       # project/allocation number;" >> ${f}
    echo "source activate torch3" >> ${f}

    echo "mpirun ${com}" >> ${f}

    echo ${f}
}

if [[ "$HOSTNAME" == *.stampede2.tacc.utexas.edu ]]; then
    f=$(gen_cluster_file ${s})

    sbatch ${f}
else
    ${com} | tee ${odir}/output.dat
fi
