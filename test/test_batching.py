#!/usr/bin/env python

from soft_patterns import Batch, SoftPatternClassifier, read_docs, read_embeddings
from util import to_cuda
import unittest
import numpy as np
import torch

from test.settings import EMBEDDINGS_FILENAME, DATA_FILENAME, MODEL_FILENAME, PATTERN_SPECS, MLP_HIDDEN_DIM, \
    NUM_MLP_LAYERS, NUM_CLASSES, SEMIRING, GPU
from util import chunked

torch.manual_seed(100)
np.random.seed(100)


class TestBatching(unittest.TestCase):
    def setUp(self):
        vocab, embeddings, word_dim = read_embeddings(EMBEDDINGS_FILENAME)
        self.embeddings = embeddings
        max_pattern_length = max(list(PATTERN_SPECS.keys()))
        self.data = read_docs(DATA_FILENAME, vocab, num_padding_tokens=max_pattern_length - 1)[0]
        state_dict = torch.load(MODEL_FILENAME)
        self.model = \
            SoftPatternClassifier(PATTERN_SPECS, MLP_HIDDEN_DIM, NUM_MLP_LAYERS, NUM_CLASSES, embeddings, vocab,
                                  SEMIRING, pre_computed_patterns, GPU)
        self.model.load_state_dict(state_dict)
        self.batch_sizes = [1, 2, 4, 5, 10, 20]

    def test_same_forward_for_diff_batches(self):
        """ Test that different batch sizes yield same `forward` results """
        # for each batch size, chunk data into batches, run model.forward,
        # then flatten results into a list (one NUM_CLASSESx1 vec per doc).
        forward_results = [
            [
                fwd
                for chunk in chunked(self.data, batch_size)
                for fwd in self.model.forward(Batch(chunk,
                                                    self.embeddings,
                                                    to_cuda(GPU))).data
            ]
            for batch_size in self.batch_sizes
        ]

        # transpose, so doc_forwards are all the diff batch sizes for a given doc
        for doc_forwards in zip(*forward_results):
            # make sure adjacent batch sizes predict the same probs
            for batch_size_a, batch_size_b in zip(doc_forwards, doc_forwards[1:]):
                for y in range(NUM_CLASSES):
                    self.assertAlmostEqual(batch_size_a[y], batch_size_b[y], places=4)


if __name__ == "__main__":
    unittest.main()
