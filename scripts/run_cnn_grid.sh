#!/usr/bin/env bash

#set -e

if [ $# -lt 2 ]; then
	echo "Usage: $0 <dataset (amazon_reviews, stanford_sentiment_binary, ROC_stories)> <n_tests> <gpu (optional)>"
	exit -1
elif [ $# -gt 2 ]; then
    gpu=$3
fi

dataset=$1
base_dir=$HOME/resources/
data_dir=$base_dir/text_cat/$dataset/

td=$data_dir/train.data
vd=$data_dir/dev.data
tl=$data_dir/train.labels
vl=$data_dir/dev.labels

e=$base_dir/glove/glove840B_text_cat_restricted.txt.gz
m=$HOME/work/soft_patterns/baselines/cnn/$dataset/

o_dir=$m/logs/
mkdir -p o_dir

function gen_cluster_file {
    local s=$1

    f=$HOME/work/soft_patterns/runs/${s}

    echo "#!/usr/bin/env bash" > ${f}
    echo "#SBATCH -J $s" >> ${f}
    echo "#SBATCH -o $o_dir/$s.out" >> ${f}
    echo "#SBATCH -p normal" >> ${f}         # specify queue
    echo "#SBATCH -N 1" >> ${f}              # Number of nodes, not cores (16 cores/node)
    echo "#SBATCH -n 1" >> ${f}
    echo "#SBATCH -t 48:00:00" >> ${f}       # max time

    echo "#SBATCH --mail-user=roysch@cs.washington.edu" >> ${f}
    echo "#SBATCH --mail-type=ALL" >> ${f}

    echo "#SBATCH -A TG-DBS110003       # project/allocation number;" >> ${f}
    echo "source activate torch3" >> ${f}

    echo "mpirun ${com}" >> ${f}

    echo ${f}
}

n=162
ind=($(seq 0 $n | sort -R))

ind2=($(seq 0 $n))

n_tests=$2

for i in $(seq $n_tests $n); do
	ind2[${ind[$i]}]=0
done

let n_tests--

for i in $(seq 0 $n_tests); do
	ind2[${ind[$i]}]=1
done

git_tag=$(git log | head -n1 | awk '{print $NF}'| cut -b -7)

i=-1
for lr in 0.01; do
	for t in 0 0.05 0.1; do
		for d in 25 50; do
		    for h in 50 100 200; do
		        for z in 4 5 6; do
		        	for c in 1 2 5; do
                    let i++
                    s=cnn_l${lr}_t${t}_d${d}_h${h}_z${z}_c${c}_${dataset}_$git_tag
                    local_d=$m/$s
                    if [ -d $local_d ]; then
                        echo "$local_d found!"
                    elif [ ${ind2[$i]} -eq 0 ]; then
#                        echo $i randomed out
			kk=22
                    else
			echo running $i
                        mkdir -p  $local_d
                        com="python -u baselines/cnn.py \
                            --td $td \
                            --vd $vd \
                            --tl $tl \
                            --vl $vl \
                            -m $local_d 	\
                            -l $lr \
                            -t $t \
                            --cnn_hidden_dim $h \
                            -i 250 \
                            -b 150 \
                            -d $d \
                            -r \
                            -x 1 \
                            -z $z \
			    --clip $c\
                            -e $e"
                        echo $com
                        if [[ "$HOSTNAME" == *.stampede2.tacc.utexas.edu ]]; then
                            f=$(gen_cluster_file ${s})
                            sbatch ${f}
                        else
                            if [ -z ${gpu+x} ]; then
                                ${com} |& tee ${o_dir}/$s.out
                            else
                                export CUDA_VISIBLE_DEVICES=$gpu && ${com} -g |& tee ${o_dir}/$s.out
                            fi
                        fi
		    fi
                done
		done
            done
        done
    done
done

exit 0
