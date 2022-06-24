EMB_FILE='glove.6B.50d.txt'
TR_DATA='data/train.data'
TR_LABELS='data/train.labels'
DEV_DATA='data/dev.data'
DEV_LABELS='data/dev.labels'
MODEL_SAVE_DIR='models_blabla/'
EPOCHS=250

MLP_HIDDEN_DIMS=(100)
DROPOUTS=(0)
LEARNING_RATES=(0.001)
declare -a PATTERNS=('5-10_4-10_3-10_2-10' '6-10_5-10_4-10' '6-10_5-10_4-10_3-10_2-10' '6-20_5-20_4-10_3-10_2-10' '7-10_6-10_5-10_4-10_3-10_2-10')
declare -a SEMIRINGS=('LogSpaceMaxTimesSemiring')
SEMIRING='MaxTimesSemiring'

COUNT=0
for MLP_DIM in "${MLP_HIDDEN_DIMS[@]}"
do
    for DOUT in "${DROPOUTS[@]}"
    do
        for LR in "${LEARNING_RATES[@]}"
        do
            for PATTERN in "${PATTERNS[@]}"
            do
                for SEMIRING in "${SEMIRINGS[@]}"
                do
                    LOGNAME="testing_patterns/${COUNT}___${MLP_DIM}_${DOUT}_${LR}___${PATTERN}.txt"
                    python3.6 ./soft_patterns.py -i $EPOCHS -d $MLP_DIM -t $DOUT -l $LR --patience 30 -e "$EMB_FILE" --td "$TR_DATA" --tl "$TR_LABELS" --vd "$DEV_DATA" --vl "$DEV_LABELS" -p "$PATTERN" --model_save_dir "$MODEL_SAVE_DIR" --semiring "$SEMIRING"  | tee $LOGNAME
                    COUNT=$(( $COUNT + 1 ))
                done
            done
        done
    done
done
