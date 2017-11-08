# soft_patterns
Neural Gappy Patterns


## Setup

```bash
    conda env create -f environment.yml
    source activate softpatterns

    datadir="/Users/sam/data/stanford_sentiment_binary" # or wherever you download the dataset
    wordvecfile="${datadir}/../glove/glove.6B.50d.txt" # e.g.
```

I also had set the following env var on Mac:

```bash
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/Users/sam/anaconda3/pkgs/mkl-11.3.3-0/lib"
```


## Training

```bash
    ./soft_patterns.py \
        -e ${wordvecfile} \
        --td ${datadir}/train.data \
        --tl ${datadir}/train.labels \
        --vd ${datadir}/dev.data \
        --vl ${datadir}/dev.labels \
        -p "5:50,4:50,3:50,2:50" \
        --maxplus
        --model_save_dir ./experiments/blah/
```

## Visualizing the Model

```bash
    ./visualize.py \
        -e ${wordvecfile} \
        --vd ${datadir}/dev.data \
        --vl ${datadir}/dev.labels \
        -p "5:50,4:50,3:50,2:50" \
        --maxplus
        --input_model ./experiments/blah/model_69.pth
```


## Running Tests

```bash
    python -m unittest
```
