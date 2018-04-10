#!/usr/bin/env bash

set -e

model_num=-1

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <output file> <model number (optional, otherwise select max dev value)>"
    exit -1
elif [ "$#" -gt 1 ]; then
    model_num=$2
fi

f=$1

function get_param {
    local f=$1
    local name=$2

    val=$(head -1 ${f} | tr ' ' '\n' | grep -E "\b${name}=" | cut -d '=' -f2 | tr -d "'()" | sed 's/,$//')

    if [ -z $val ] || [ $val == 'False' ]; then
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
    while [ ! -e ${model_dir}/model_${model_num}.pth ]; do
	model_num=$(grep loss $f | awk '{print $2" "$NF}' | sort -rnk2 | sed "${i}q;d" | awk '{print $1}')
	let i++
    done
fi

if [ ${model_num} -eq -1 ]; then
	echo "Couldn't find model file in $model_dir"
	exit -2
fi


e=$(get_param ${f} embedding_file)
maxplus=$(get_param ${f} maxplus)
maxtimes=$(get_param ${f} maxtimes)
echo $maxtimes is mt
mlp_hidden_dim=$(get_param ${f} mlp_hidden_dim)
num_mlp_layers=$(get_param ${f} num_mlp_layers)
patterns=$(get_param ${f} patterns)
no_eps=$(get_param ${f} no_eps)
no_sl=$(get_param ${f} no_sl)
seed=$(get_param ${f} seed)
vd=$(get_param ${f} td | awk '{print $2}')
vl=$(get_param ${f} tl | awk '{print $2}')


com="python -u visualize_efficiently.py  \
    ${e} \
    ${patterns} \
    --vd ${vd} \
    --vl ${vl} \
    $seed \
    ${mlp_hidden_dim} \
    ${maxplus} \
    ${maxtimes} \
    -b 150 \
    -k 10 $no_sl $no_eps \
    ${num_mlp_layers} \
    --input_model ${model_dir}/model_${model_num}.pth"

echo ${com}

of=${model_dir}/viz_${model_num}.dat

if [ -e $of ]; then
	echo "$of found. Please remove it if you want to override it"
	exit -2
fi 

${com} | tee $of
