#!/usr/bin/env bash

set -e

model_num=-1

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <output file> <model: 0 for soft_patterns, 1 for dan, 2 for lstm, 3 for cn> <model number (or -1 for best)> <test data> <test labels> <gpu num (or -1)>"
    exit -1
fi

f=$1
model=$2
model_num=$3
test_data=$4
test_labels=$5
gpu=$6

n=$(echo $f | awk -F / '{print $NF}')

function get_param {
    local f=$1
    local name=$2

    val=$(head -1 ${f} | tr ' ' '\n' | grep -E "\b${name}=" | cut -d '=' -f2 | tr -d "'()" | sed 's/,$//')

    if [ -z $val ] || [ $val == 'False' ] || [ $val == 'None' ]; then
	echo ""
    elif [ $val == 'True' ]; then 
	echo "--$name"
    else
	echo --$name ${val}
    fi
}

model_dir=$(get_param $f model_save_dir | awk '{print $2}')

if [ ${model_num} -eq -1 ]; then
    i=1
    while [ ! -e ${model_dir}/model_${model_num}.pth ];
    do
	model_num=$(grep loss $f | awk '{print $2" "$NF}' | sort -rnk2 | sed "${i}q;d" | awk '{print $1}')
	let i++
    done
fi

if [ ${model_num} -eq -1 ]; then
	echo "Couldn't find model file in $model_dir"
	exit -2
fi

if [ $gpu -eq -1 ]; then
    gpu_flag=''
else
    gpu_flag='--gpu'
fi


e=$(get_param ${f} embedding_file)
seed=$(get_param ${f} seed)
mlp_hidden_dim=$(get_param ${f} mlp_hidden_dim)
num_mlp_layers=$(get_param ${f} num_mlp_layers)
hidden_dim=$(get_param ${f} hidden_dim)


if [ $model -eq 0 ]; then
    maxplus=$(get_param ${f} maxplus)
    maxtimes=$(get_param ${f} maxtimes)

    patterns=$(get_param ${f} patterns)
    rnn=$(get_param ${f} use_rnn)
    no_sl=$(get_param ${f} no_sl)
    no_eps=$(get_param ${f} no_eps)
    eps_scale=$(get_param ${f} eps_scale)
    self_loop_scale=$(get_param ${f} self_loop_scale)
    bias_scale_param=$(get_param ${f} bias_scale_param)

    shared_sl=$(get_param ${f} shared_sl)

    s="$patterns $maxplus $maxtimes $no_sl $no_eps $shared_sl $bias_scale_param $eps_scale $self_loop_scale"
elif [ $model -eq 1 ]; then
    s="--dan"
elif [ $model -eq 2 ]; then
    s="--bilstm"
elif [ $model -eq 3 ]; then
    num_cnn_layers=$(get_param ${f} num_cnn_layers)
    cnn_hidden_dim=$(get_param ${f} cnn_hidden_dim)
    window_size=$(get_param ${f} window_size)
    s="--cnn $num_cnn_layers $cnn_hidden_dim $window_size"
else
    echo "Model not found (should be 0, 1 or 3. Got $model"
    exit -3
fi


com="python -u soft_patterns_test.py  \
    ${e} \
    --vd ${test_data} \
    --vl ${test_labels} \
    ${mlp_hidden_dim} \
    $seed \
    -b 150 $gpu_flag \
    $s \
    $rnn \
    $hidden_dim \
    ${num_mlp_layers} \
    --input_model ${model_dir}/model_${model_num}.pth"

echo ${com}


function gen_cluster_file {
    local n=$1

    f=$HOME/work/soft_patterns/test_runs/${n}

    echo "#!/usr/bin/env bash" > ${f}
    echo "#SBATCH -J test_$n" >> ${f}
    echo "#SBATCH -o $HOME/work/soft_patterns/test_results/$n" >> ${f}
    echo "#SBATCH -p normal" >> ${f}         # specify queue
    echo "#SBATCH -N 1" >> ${f}              # Number of nodes, not cores (16 cores/node)
    echo "#SBATCH -n 1" >> ${f}
    echo "#SBATCH -t 01:00:00" >> ${f}       # max time

    echo "#SBATCH --mail-user=roysch@cs.washington.edu" >> ${f}
    echo "#SBATCH --mail-type=ALL" >> ${f}

    echo "#SBATCH -A TG-DBS110003       # project/allocation number;" >> ${f}
    echo "source activate torch3" >> ${f}

    echo "mpirun ${com}" >> ${f}

    echo ${f}
}

if [[ "$HOSTNAME" == *.stampede2.tacc.utexas.edu ]]; then
    f=$(gen_cluster_file ${n})

    sbatch ${f}
elif [ $gpu -eq -1 ]; then
    ${com} | tee ${model_dir}/test_results.dat
else
    export CUDA_VISIBLE_DEVICES=$gpu &&  ${com} | tee ${model_dir}/test_results.dat
fi

