#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model dir> <model number (optional, otherwise select max dev value)>"
elif [ "$#" -gt 1 ]; then
    model_num=$2
else
    model_num=$(grep loss $1/output.dat | awk '{print $2" "$NF}' | sort -rnk2 | head -1 | awk '{print $1}')
fi

model_dir=$1

params=$(head -1 $model_dir/output.dat)

function get_param {
    local str=$1
    local name=$2

    return $(echo $str | tr ' ' '\n' | grep $name | cut -d '=' -f2 | tr -d ",'")
}

e=get_param $params embedding_file
maxplus=get_param $params maxplus

if [ $maxplus == 'True' ]; then
    maxplus="--maxplus"
else
    maxplus=''
fi


mlp_hidden_dim=get_param $params mlp_hidden_dim
num_mlp_layers=get_param $params num_mlp_layers
patterns=get_param $params patterns
vd=get_param $params vd
vl=get_param $params vl

com="python -u soft_patterns.py  \
    -e $e \
    -p $patterns \
     --vd $vd \
     --vl $vl \
     -d $mlp_hidden_dim $maxplus \
     --num_mlp_layers $num_mlp_layers \
     -m $model_dir/model_$model_num.pth"

#$com | tee $model_dir/viz_$model_num.dat

