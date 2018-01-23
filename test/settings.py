from collections import OrderedDict

from soft_patterns import MaxPlusSemiring

DATA_FILENAME = 'test/data/train.20examples.data'
EMBEDDINGS_FILENAME = 'test/data/glove.6B.50d.20words.txt'
MODEL_FILENAME = 'test/data/model.pth'
PATTERN_SPECS = OrderedDict([int(y) for y in x.split(":")] for x in "5:50".split(","))
NUM_MLP_LAYERS = 2
NUM_CLASSES = 2
MLP_HIDDEN_DIM = 10
WINDOW_SIZE = 4
NUM_CNN_LAYERS = 1
CNN_HIDDEN_DIM = 10
LSTM_HIDDEN_DIM = 10
SEMIRING = MaxPlusSemiring
GPU = False
