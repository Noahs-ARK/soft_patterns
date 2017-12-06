#!/usr/bin/env bash

set -e

model_num=-1

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <output file> <model: 0 for soft_patterns, 1 for dan, 2 for lstm> <model number> <test data> <test labels>"
    exit -1
fi

f=$1
model=$2
model_num=$3
test_data=$4
test_labels=$5

function get_param {
    local f=$1
    local name=$2

    val=$(head -1 ${f} | tr ' ' '\n' | grep -E "\b${name}=" | cut -d '=' -f2 | tr -d "'()" | sed 's/,$//')
    echo ${val}
}

model_dir=$(get_param $f model_save_dir)

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


e=$(get_param ${f} embedding_file)
gpu=$(get_param ${f} gpu)
seed=$(get_param ${f} seed)
mlp_hidden_dim=$(get_param ${f} mlp_hidden_dim)
num_mlp_layers=$(get_param ${f} num_mlp_layers)

if [ "${gpu}" == 'True' ]; then
    gpu='--gpu'
else
    gpu=''
fi

if [ $model -eq 0 ]; then
    maxplus=$(get_param ${f} maxplus)

    if [ "${maxplus}" == 'True' ]; then
        maxplus="--maxplus"
    else
        maxtimes=$(get_param ${f} maxtimes)
        if [ "${maxtimes}" == 'True' ]; then
            maxplus="--maxtimes"
        else
            maxplus=''
        fi
    fi
    patterns=$(get_param ${f} patterns)

    s="-p $patterns $maxplus"
elif [ $model -eq 1 ]; then
    s="--dan"
elif [ $model -eq 2 ]; then
    lstm_hidden=$(get_param ${f} hidden_dim)
    s="--bilstm --hidden_bilstm_dim $lstm_hidden"
else
    echo "Model not found (should be 0, 1 or 2. Got $model"
    exit -3
fi


com="python -u soft_patterns_test.py  \
    -e ${e} \
    --vd ${test_data} \
    --vl ${test_labels} \
    --td '' \
    --tl '' \
    -d ${mlp_hidden_dim} \
    --seed $seed \
    -b 150 $gpu \
    $s \
    --num_mlp_layers ${num_mlp_layers} \
    --input_model ${model_dir}/model_${model_num}.pth"

echo ${com}

${com} | tee ${model_dir}/test_results.dat
