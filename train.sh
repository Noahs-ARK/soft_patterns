# Use this instead of having to copy/paste into shell each time
EMB_FILE='<word_embeddings_file>'
TR_DATA='data/train.data'
TR_LABELS='data/train.labels'
DEV_DATA='data/dev.data'
DEV_LABELS='data/dev.labels'
PATTERN='<pattern specification>'
MODEL_SAVE_DIR='<output model directory>'

python3.6 ./soft_patterns.py -e "$EMB_FILE" --td "$TR_DATA" --tl "$TR_LABELS" --vd "$DEV_DATA" --vl "$DEV_LABELS" -p "$PATTERN" --model_save_dir "$MODEL_SAVE_DIR"