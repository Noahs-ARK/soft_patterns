# Use this instead of having to copy/paste into shell each time
EMB_FILE='test/data/glove.6B.50d.20words.txt'
TR_DATA='data/train.data'
TR_LABELS='data/train.labels'
DEV_DATA='data/dev.data'
DEV_LABELS='data/dev.labels'
PATTERN='5-50_4-50_3-50_2-50' # "Pattern lengths and numbers: an underscore separated list of length-number pairs (default: 5-50_4-50_3-50_2-50)". Taken from one of the files
MODEL_SAVE_DIR='models_blabla/'

python3.6 ./soft_patterns.py -e "$EMB_FILE" --td "$TR_DATA" --tl "$TR_LABELS" --vd "$DEV_DATA" --vl "$DEV_LABELS" -p "$PATTERN" --model_save_dir "$MODEL_SAVE_DIR"