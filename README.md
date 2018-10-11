# Soft Patterns
Text classification code using SoPa, based on ["SoPa: Bridging CNNs, RNNs, and Weighted Finite-State Machines"](https://arxiv.org/abs/1805.06061) by Roy Schwartz, Sam Thomson and Noah A. Smith, ACL 2018


## Setup

The code is implemented in python3.6 using pytorch. To run, we recommend using conda. The following code creates a new conda environment and activates it:

```bash
./install.sh
source activate sopa
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

### Sample data
The ```data/``` folder contains sample files for training, development and testing. 
The data comes from the SST dataset (with a 100 training samples).

Each fold X (train, dev, test) contains two file: X.data (plain text sentences, one sentence per line) and X.labels (one label per line).


## Visualizing the Model
Under construction.

## Sanity Tests

```bash
python -m unittest
```

## References
If you make use if this code, please cite the following paper:

```bash

@inproceedings{Schwartz:2018,
  author={Schwartz, Roy and Thomson, Sam and Smith, Noah A.},
  title={{SoPa}: Bridging {CNNs}, {RNNs}, and Weighted Finite-State Machines},
  booktitle={Proc. of ACL},
  year={2018}
}
```

## Contact

For questions, comments or feedback, please email roysch@cs.washington.edu
