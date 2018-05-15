# Soft Patterns
Text classification code using SoPa, based on ["SoPa: Bridging CNNs, RNNs, and Weighted Finite-State Machines"](https://arxiv.org/) by Roy Schwartz, Sam Thomson and Noah A. Smith, ACL 2018


## Setup

The code is implemented in python3.6 using pytorch. To run, we recommend using conda:

```bash
conda env create -f environment.yml
source activate softpatterns
```

### Data format
The training and test code requires a two files for training, development and test: a data file and a labels file.
Both files contain one line per sample. The data file contains the text, and the labels file contain the label.
In addition, a word vector file is required (plain text, standard format of one line per vector, starting with the word, followed by the vector).

For other paramteres, run the following commands using the ```--help``` flag.

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

```bash
python3.6 ./soft_patterns_test.py \
    -e <word embeddings file> \
    --vd <test data> \
    --vl <test labels> \
    -p <pattern specification> \
    --input_model <input model>
```


### TODO

1. fix visualization script
2. train doesn't run with gpu -- resolved(cuda version issue)
3. test doesn't correspond to train -- resolved
4. unittest fails
5. Address interpret_classification_results.py
6. Make sure interpret_classification_results.py works

## Visualizing the Model
SoPa offers two types of visualization. One
```bash
python3.6 ./visualize.py \
    --vd <data file to visualize> \
    --vl <labels of data file to visualize> \
    --input_model <input model>
    -p <pattern specification> \
    --input_model <input model>"
```


## Sanity Tests

```bash
python -m unittest
```
