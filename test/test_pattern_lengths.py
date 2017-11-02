#!/usr/bin/env python

import unittest

import numpy as np
import torch
from torch.autograd import Variable

from data import read_embeddings, read_docs
from soft_patterns import fixed_var, SoftPatternClassifier, Batch
from test.settings import EMBEDDINGS_FILENAME, DATA_FILENAME, MODEL_FILENAME, PATTERN_SPECS, MLP_HIDDEN_DIM, \
    NUM_MLP_LAYERS, NUM_CLASSES, SEMIRING, GPU, DROPOUT, LEGACY

torch.manual_seed(100)
np.random.seed(100)


def forward(model, batch):
    """ old version, for reference """
    transition_matrices = model.get_transition_matrices(batch)
    scores = Variable(model.semiring.zero(batch.size(), model.total_num_patterns).type(model.dtype))

    # to add start state for each word in the document.
    restart_padding = fixed_var(model.semiring.one(model.total_num_patterns, 1), model.gpu)
    zero_padding = fixed_var(model.semiring.zero(model.total_num_patterns, 1), model.gpu)
    eps_value = \
        model.semiring.times(
            model.semiring.from_float(model.epsilon_scale),
            model.semiring.from_float(model.epsilon)
        )
    self_loop_scale = model.get_self_loop_scale()

    # Different documents in batch
    for doc_index in range(len(transition_matrices)):
        # Start state
        hiddens = Variable(model.semiring.zero(model.total_num_patterns, model.max_pattern_length).type(model.dtype))
        hiddens[:, 0] = model.semiring.one(model.total_num_patterns, 1).type(model.dtype)
        # For each token in document
        for transition_matrix_val in transition_matrices[doc_index]:
            hiddens = model.transition_once(eps_value,
                                            hiddens,
                                            self_loop_scale,
                                            transition_matrix_val,
                                            zero_padding,
                                            restart_padding)
            # Score is the final column of hiddens
            start = 0
            for pattern_len, num_patterns in model.pattern_specs.items():
                end_state = -1 - (model.max_pattern_length - pattern_len)
                end_pattern_idx = start + num_patterns
                scores[doc_index, start:end_pattern_idx] = \
                    model.semiring.plus(
                        scores[doc_index, start:end_pattern_idx],
                        hiddens[start:end_pattern_idx, end_state]
                    )  # mm(hidden, self.final)  # TODO: change if learning final state
                start += num_patterns

    return model.mlp.forward(scores)


class TestPatternLengths(unittest.TestCase):
    def setUp(self):
        vocab, embeddings, word_dim = read_embeddings(EMBEDDINGS_FILENAME)
        self.embeddings = embeddings
        self.data = read_docs(DATA_FILENAME, vocab)[0]
        state_dict = torch.load(MODEL_FILENAME)
        self.model = \
            SoftPatternClassifier(
                PATTERN_SPECS,
                MLP_HIDDEN_DIM,
                NUM_MLP_LAYERS,
                NUM_CLASSES,
                embeddings,
                vocab,
                SEMIRING,
                GPU,
                DROPOUT,
                LEGACY
            )
        self.model.load_state_dict(state_dict)

    def test_pattern_lengths(self):
        """
        Test that using `torch.gather` for collecting end-states works the
        same as doing it manually
        """
        batch = Batch(self.data, self.embeddings, GPU)
        expected = forward(self.model, batch).data
        actual = self.model.forward(batch).data
        for expd_doc, act_doc in zip(expected, actual):
            for expd_y, act_y in zip(expd_doc, act_doc):
                self.assertAlmostEqual(expd_y, act_y, places=4)


if __name__ == "__main__":
    unittest.main()
