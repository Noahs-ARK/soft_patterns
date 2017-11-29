#!/usr/bin/env bash

set -e

model_num=-1

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <output file> <model number> <test data> <test labels>"
    exit -1
fi

f=$1
model_num=$2
test_data=$3
test_labels=$4

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
maxplus=$(get_param ${f} maxplus)
seed=$(get_param ${f} seed)

if [ "${gpu}" == 'True' ]; then
    gpu='--gpu'
else
    gpu=''
fi

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


mlp_hidden_dim=$(get_param ${f} mlp_hidden_dim)
num_mlp_layers=$(get_param ${f} num_mlp_layers)
patterns=$(get_param ${f} patterns)


com="python -u soft_patterns_test.py  \
    -e ${e} \
    -p ${patterns} \
    --vd ${test_data} \
    --vl ${test_labels} \
    --td '' \
    --tl '' \
    -d ${mlp_hidden_dim} \
    ${maxplus} \
    --seed $seed \
    -b 150 $gpu \
    --num_mlp_layers ${num_mlp_layers} \
    --input_model ${model_dir}/model_${model_num}.pth"

echo ${com}

${com} | tee ${model_dir}/test_results.dat
