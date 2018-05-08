# Soft Patterns
Text classification code using SoPa, based on ["SoPa: Bridging CNNs, RNNs, and Weighted Finite-State Machines"](https://arxiv.org/) by Roy Schwartz, Sam Thomson and Noah A. Smith, ACL 2018


## Setup

The code is implemented in python3.6 using pytorch. To run, we recommend using conda:

```bash
conda env create -f environment.yml
source activate softpatterns```

### Data format
The training and test code requires a two files for training, development and test: a data file and a labels file.
Both files contain one line per sample. The data file contains the text, and the labels file contain the label.
In addition, a word vector file is required (plain text, standard format of one line per vector, starting with the word, followed by the vector).

For other paramteres, run the following commands using the ````--help``` flag.

## Training

To train our model, run

```bash
python3.6 ./soft_patterns.py \
    -e <word embeddings file> \
    --td <train data> \
    --tl <train labels> \
    --vd <dev data> \
    --vl <dev labels> \
    -p <pattern specification> \
    --model_save_dir <output model directory>
```

### Test
To test our model, run

```bash python3.6 ./soft_patterns_test.py \
    -e <word embeddings file> \
    --vd <test data> \
    --vl <test labels> \
    -p <pattern specification> \
    --input_model <input model>
```


### TODO

1. fix visualization script
2. train doesn't run with gpu
3. test doesn't correspond to train
4. unittest fails

## Visualizing the Model
SoPa offers two types of visualization. One
```bash python3.6 ./visualize.py \
    --vd <data file to visualize> \
    --vl <labels of data file to visualize> \
    --input_model <input model>
    -p "5-10_4-10_3-20_2-20" \
    -b 1000 \
    --maxtimes \
    --input_model "${model_dir}/output_p5-10_4-10_3-20_2-20_d1_l1e-3_t0.2_r_b_6B.100d_slScale0_epsScale0_3d79c4f/model_25.pth"
```


## Sanity Tests

```bash
python -m unittest
```
