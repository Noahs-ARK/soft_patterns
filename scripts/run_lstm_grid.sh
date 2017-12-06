#!/usr/bin/env bash

#set -e

if [ $# -lt 1 ]; then
	echo "Usage: $0 <dataset (amazon_reviews, stanford_sentiment_binary, ROC_stories)>"
	exit -1
fi

dataset=$1
base_dir=$HOME/resources/
data_dir=$base_dir/text_cat/$dataset/

td=$data_dir/train.data
vd=$data_dir/dev.data
tl=$data_dir/train.labels
vl=$data_dir/dev.labels

e=$base_dir/glove/glove.840B.300d.txt
m=$HOME/work/soft_patterns/baselines/lstm/$dataset/

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

n=107
ind=($(seq 0 $n | sort -R))

ind2=($(seq 0 $n))

for i in {0..15}; do
	ind2[${ind[$i]}]=1
done

for i in $(seq 16 $n); do
	ind2[${ind[$i]}]=0
done

git_tag=$(git log | head -n1 | awk '{print $NF}'| cut -b -7)

i=-1
for lr in 0.01 0.005 0.001; do
	for t in 0.05 0.1 0.2; do
		for d in 10 25 50 100; do
		    for h in 100 200 300; do
			let i++
        	        s=lstm_l${lr}_t${t}_d${d}_h${h}_${dataset}_$git_tag
                	local_d=$m/$s
			if [ -d $local_d ]; then
			    echo "$local_d found!"
			elif [ ${ind2[$i]} -eq 0 ]; then
			    echo $i randomed out
			else	
		                mkdir -p  $local_d
	        	        com="python -u baselines/lstm.py \
		                    --td $td \
        		            --vd $vd \
                		    --tl $tl \
		                    --vl $vl \
        		            -m $local_d 	\
                	 	    -l $lr \
		                    -t $t \
        		            --hidden_dim $h \
                		    -i 250 \
	                	    -b 150 \
	        	            -d $d \
        	        	    -r \
	        	            -e $e"
	        	        echo $com

				if [[ "$HOSTNAME" == *.stampede2.tacc.utexas.edu ]]; then
				    f=$(gen_cluster_file ${s})
				    sbatch ${f}
				else
				    ${com} |& tee ${local_dir}/output.dat
				fi
			fi
        	    done
		done
	done
done

exit 0
