#!/usr/bin/env python

from soft_patterns import Batch, MaxPlusSemiring, SoftPatternClassifier, read_docs, read_embeddings
import unittest
from collections import OrderedDict
import numpy as np
import torch

torch.manual_seed(100)
np.random.seed(100)

DATA_FILENAME = 'test/data/train.20examples.data'
EMBEDDINGS_FILENAME = 'test/data/glove.6B.50d.20words.txt'
MODEL_FILENAME = 'test/data/model.pth'
PATTERN_SPECS = OrderedDict([int(y) for y in x.split(":")] for x in "5:50".split(","))
NUM_MLP_LAYERS = 2
NUM_CLASSES = 2
MLP_HIDDEN_DIM = 10
SEMIRING = MaxPlusSemiring
GPU = False
DROPOUT = 0.2
LEGACY = False


class TestBatching(unittest.TestCase):
    def test_batching(self):
        """ Test that different batch sizes yield same `forward` results """
        vocab, reverse_vocab, embeddings, word_dim = \
            read_embeddings(EMBEDDINGS_FILENAME)
        data, _ = read_docs(DATA_FILENAME, vocab)
        state_dict = torch.load(MODEL_FILENAME)
        model = \
            SoftPatternClassifier(
                PATTERN_SPECS,
                MLP_HIDDEN_DIM,
                NUM_MLP_LAYERS,
                NUM_CLASSES,
                embeddings,
                reverse_vocab,
                SEMIRING,
                GPU,
                DROPOUT,
                LEGACY
            )
        model.load_state_dict(state_dict)

        batch_sizes = [1, 2, 4, 5, 10, 20]

        results = []
        for batch_size in batch_sizes:
            batches = [
                Batch(data[i:i + batch_size], embeddings, GPU)
                for i in range(0, len(data), batch_size)
            ]
            forwards = [
                fwd.data
                for batch in batches
                for fwd in model.forward(batch)
            ]
            results.append(forwards)

        for fwd in zip(*results):
            for a, b in zip(fwd, fwd[1:]):
                self.assertAlmostEqual(a[0], b[0], delta=0.05)
                self.assertAlmostEqual(a[1], b[1], delta=0.05)


if __name__ == "__main__":
    unittest.main()
