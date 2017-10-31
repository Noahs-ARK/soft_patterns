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

        # for each batch size, chunk data into batches, run model.forward,
        # then flatten results into a list (one NUM_CLASSESx1 vec per doc).
        forward_results = [
            [
                fwd
                for start_idx in range(0, len(data), batch_size)
                for fwd in model.forward(Batch(data[start_idx:start_idx + batch_size],
                                               embeddings,
                                               GPU)).data
            ]
            for batch_size in batch_sizes
        ]

        # transpose, so doc_forwards are all the diff batch sizes for a given doc
        for doc_forwards in zip(*forward_results):
            # make sure adjacent batch sizes predict the same probs
            for a, b in zip(doc_forwards, doc_forwards[1:]):
                for y in range(NUM_CLASSES):
                    self.assertAlmostEqual(a[y], b[y], delta=0.05)


if __name__ == "__main__":
    unittest.main()
