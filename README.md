# soft_patterns
Neural Gappy Patterns


## Setup

    conda env create -f environment.yml
    source activate softpatterns

I also had set the following env var on Mac:

    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/Users/sam/anaconda3/pkgs/mkl-11.3.3-0/lib"

## Training

    datadir="/Users/sam/data/stanford_sentiment_binary" # or wherever you download the dataset
    wordvecfile="${datadir}/../glove/glove.6B.50d.txt" # e.g.

    python3 soft_patterns.py \
        -e ${wordvecfile} \
        --td ${datadir}/train.data \
        --tl ${datadir}/train.labels \
        --vd ${datadir}/dev.data \
        --vl ${datadir}/dev.labels \
        -p "5:50,4:50,3:50,2:50" \
        --model_save_dir ./experiments/blah/


## Running Tests

    python -m unittest test.forward_one_sent
