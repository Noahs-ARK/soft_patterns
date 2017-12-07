#!/usr/bin/env  bash

dirs=(stanford_sentiment_binary amazon_reviews ROC_stories stanford_sentiment_binary_100 stanford_sentiment_binary_500 stanford_sentiment_binary_1000 stanford_sentiment_binary_2500 amazon_reviews_100 amazon_reviews_500 amazon_reviews_1000 amazon_reviews_2500 amazon_reviews_5000 amazon_reviews_10000)

n_dirs=${#dirs[@]}
let n_dirs--

if [ $# -lt 3 ]; then
	echo "Usage: $0 <dataset> <seed> <gpu>"
	echo "Dirs are:"
	for i in $(seq 0 $n_dirs); do
		echo ${i}: ${dirs[$i]}
	done
	exit -1
fi

#set -e

dir=${dirs[$1]}

ps=(6:20,5:20,4:10,3:10,2:10 6:10,5:10,4:10,3:10,2:10 7:10,6:10,5:10,4:10,3:10,2:10 6:10,5:10,4:10 5:10,4:10,3:10,2:10)
#ls=(0.01 0.005 0.001)
ls=(0.01 0.05 0.01 0.005)
ws=(0 0.05 0.1 0.15)
#ws=(0 0.05 0.1)
#ts=(0 0.05 0.1)
ts=(0 0.05 0.1 0.2 0.3)
ds=(0 10 25 50)
n=$(echo "${#ps[@]} * ${#ls[@]} * ${#ts[@]} * ${#ds[@]} * ${#ws[@]}" | bc -l)
hs=(100 200 300)
echo $n
ind=($(seq 0 $n | sort -R))

ind2=($(seq 0 $n))

for i in {0..49}; do
	ind2[${ind[$i]}]=1
done

for i in $(seq 50 $n); do
	ind2[${ind[$i]}]=0
done



i=-1
for p in ${ps[@]}; do
        for l in ${ls[@]}; do
                for t in ${ts[@]}; do
                        for d in ${ds[@]}; do
                        	for w in ${ws[@]}; do
                            	for h in ${hs[@]}; do
				                	let i++
					                p2=$(echo ${p} | tr ',' '_' | tr ':' '-')
                	                s=$HOME/work/soft_patterns/output_p${p2}_d${d}_l${l}_t${t}_r_mt_b150_clip0_840B.300d_w${w}_${dir}_seed${2}_bh$h
	
                                    if [ ${ind2[$i]} -eq 0 ]; then
                                        echo $i randomed out
                                        continue
                                    fi

                #					ls ${s}*/output.dat
                                    v=$(ls ${s}*/output.dat |& grep 'No such file' | wc -l)

                                    if [ $v -gt 0 ]; then
                #						echo run
                #						exit -1
                                        export CUDA_VISIBLE_DEVICES=$3 && ./scripts/run_code.sh $p $d $l $t 1 2 150  0 1 2 0 $w $1 $2 $h
                                    fi
			                	done
			                done
                        done
                done
        done
done
