# Use this instead of having to copy/paste into shell each time
EMB_FILE='test/data/glove.840B.300d.txt' #glove.840B.300d #glove.6B.50d.20words
PATTERN='7-10_6-10_5-10_4-10_3-10_2-10' # "Pattern lengths and numbers: an underscore separated list of length-number pairs (default: 5-50_4-50_3-50_2-50)". Taken from one of the files
INPUT_MODEL='models_blabla/model_0.pth'
TEST_DATA='data/test.data'
TEST_LABELS='data/test.labels'
MLP_DIM=100

python3.6 ./soft_patterns_test.py \
    -d $MLP_DIM \
    -e "$EMB_FILE" \
    --vd "$TEST_DATA" \
    --vl "$TEST_LABELS" \
    -p "$PATTERN" \
    --input_model "$INPUT_MODEL"\
    -- no_sl