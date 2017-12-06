# soft_patterns
Neural Gappy Patterns


## Setup

```bash
conda env create -f environment.yml
source activate softpatterns

export data_dir=~/data
export model_dir=~/code/soft_patterns/experiments
sst_dir="${data_dir}/text_cat/stanford_sentiment_binary"  # or wherever you download the dataset
wordvec_file="${data_dir}/glove/glove.6B.50d.txt" # e.g.
```

I also had set the following env var on Mac:

```bash
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/Users/sam/anaconda3/pkgs/mkl-11.3.3-0/lib"
```


## Training

```bash
./soft_patterns.py \
    -e ${wordvec_file} \
    --td ${sst_dir}/train.data \
    --tl ${sst_dir}/train.labels \
    --vd ${sst_dir}/dev.data \
    --vl ${sst_dir}/dev.labels \
    -p "5:10,4:10,3:20,2:20" \
    --mlp_hidden_dim 10 \
    --maxtimes \
    --learning_rate 1e-3 \
    --dropout 0.1 \
    --scheduler \
    --model_save_dir "${model_dir}/output_p5-10_4-10_3-20_2-20_d1_l1e-3_t0.2_r_b_6B.100d_slScale0_epsScale0_3d79c4f"
```

or

```bash
./run_code.sh \
    5:50,4:50,3:50,2:50 \
    10 \
    1e-3 \
    0.1 \
    1 \
    1
```

## Visualizing the Model

```bash
./visualize.py \
    -e "${wordvec_file}" \
    --vd "${sst_dir}/dev.data" \
    --vl "${sst_dir}/dev.labels" \
    -p "5:10,4:10,3:20,2:20" \
    -b 1000 \
    --maxtimes \
    --input_model "${model_dir}/output_p5-10_4-10_3-20_2-20_d1_l1e-3_t0.2_r_b_6B.100d_slScale0_epsScale0_3d79c4f/model_25.pth"
```

or

```bash
./run_viz.sh \
    "${model_dir}/output_p5-10_4-10_3-20_2-20_d1_l1e-3_t0.2_r_b_6B.100d_slScale0_epsScale0_3d79c4f"
```


## Running Tests

```bash
python -m unittest
```
