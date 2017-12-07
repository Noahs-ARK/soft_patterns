#!/usr/bin/env python

import unittest

import numpy as np
import torch
from torch import cat, mm, FloatTensor
from torch.autograd import Variable
import soft_patterns
from data import read_embeddings, read_docs, Vocab
from soft_patterns import fixed_var, SoftPatternClassifier
from util import to_cuda
from test.settings import EMBEDDINGS_FILENAME, DATA_FILENAME, MODEL_FILENAME, PATTERN_SPECS, MLP_HIDDEN_DIM, \
    NUM_MLP_LAYERS, NUM_CLASSES, SEMIRING, GPU

torch.manual_seed(100)
np.random.seed(100)


def forward(model, batch):
    """ old version, for reference """
    transition_matrices = get_transition_matrices(model, batch)
    scores = Variable(model.semiring.zero(batch.size(), model.total_num_patterns))

    # to add start state for each word in the document.
    restart_padding = fixed_var(model.semiring.one(model.total_num_patterns, 1))
    zero_padding = fixed_var(model.semiring.zero(model.total_num_patterns, 1))
    eps_value = \
        model.semiring.times(
            model.semiring.from_float(model.epsilon_scale),
            model.semiring.from_float(model.epsilon)
        )
    all_hiddens = []

    # Different documents in batch
    for doc_index in range(len(transition_matrices)):
        # Start state
        hiddens = Variable(model.semiring.zero(model.total_num_patterns, model.max_pattern_length))
        hiddens[:, 0] = model.semiring.one(model.total_num_patterns, 1)
        all_hiddens.append(hiddens)
        # For each token in document
        for transition_matrix_val in transition_matrices[doc_index]:
            hiddens = transition_once(model,
                                      eps_value,
                                      hiddens,
                                      model.self_loop_scale,
                                      transition_matrix_val,
                                      zero_padding,
                                      restart_padding)
            all_hiddens.append(hiddens)
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

    return model.mlp.forward(scores), transition_matrices, all_hiddens


def get_transition_matrices(model, batch):
    mm_res = mm(model.diags, batch.embeddings_matrix)
    transition_probs = \
        model.semiring.from_float(mm_res + model.bias.expand(model.bias.size()[0], mm_res.size()[1])).t()

    # transition matrix for each document in batch
    transition_matrices = [
        [
            transition_probs[word_index, :].contiguous().view(
                model.total_num_patterns, model.num_diags, model.max_pattern_length
            )
            for word_index in doc
        ]
        for doc in batch.docs
    ]
    return transition_matrices


def transition_once(model,
                    eps_value,
                    hiddens,
                    self_loop_scale,
                    transition_matrix_val,
                    zero_padding,
                    restart_padding):
    # Adding epsilon transitions (don't consume a token, move forward one state)
    # We do this before self-loops and single-steps.
    # We only allow one epsilon transition in a row.
    hiddens = \
        model.semiring.plus(
            hiddens,
            cat((zero_padding,
                 model.semiring.times(
                     hiddens[:, :-1],
                     eps_value  # doesn't depend on token, just state
                 )), 1))
    # single steps forward (consume a token, move forward one state)
    # print(hiddens[:, -1])
    # print("RESTART old: ", restart_padding)

    # print("ef", restart_padding.size(), hiddens[:, -1].size(), transition_matrix_val[:, 1, :-1].size())
    result = \
        cat((restart_padding,  # <- Adding the start state
             model.semiring.times(
                 hiddens[:, :-1],
                 transition_matrix_val[:, 1, :-1])
             ), 1)
    # Adding self loops (consume a token, stay in same state)
    result = \
        model.semiring.plus(
            result,
            model.semiring.times(
                self_loop_scale,
                model.semiring.times(
                    hiddens,
                    transition_matrix_val[:, 0, :]
                )
            )
        )
    return result


class TestPatternLengths(unittest.TestCase):
    def setUp(self):
        vocab, embeddings, word_dim = read_embeddings(EMBEDDINGS_FILENAME)
        self.embeddings = embeddings
        max_pattern_length = max(list(PATTERN_SPECS.keys()))
        self.data = read_docs(DATA_FILENAME, vocab, 0)[0]
        state_dict = torch.load(MODEL_FILENAME)
        self.model = \
            SoftPatternClassifier(PATTERN_SPECS, MLP_HIDDEN_DIM, NUM_MLP_LAYERS, NUM_CLASSES, embeddings, vocab,
                                  SEMIRING, pre_computed_patterns, GPU)
        self.model.load_state_dict(state_dict)

    def test_pattern_lengths(self):
        """
        Test that using `torch.gather` for collecting end-states works the
        same as doing it manually
        """
        test_data = [self.data[0]]
        batch = Batch(test_data, self.embeddings)
        batch2 = soft_patterns.Batch(test_data, self.embeddings, to_cuda(GPU))
        expected, transition_expected, all_hiddens_expected = forward(self.model, batch)
        actual, transition_actual, all_hiddens_actual = self.model.forward(batch2, 3)

        for mat_actual, mat_expected in zip(transition_actual, transition_expected):
            for i in range(mat_actual.size()[1]):
                for j in range(mat_actual.size()[2]):
                    for k in range(mat_actual.size()[3]):
                        k1 = mat_actual[0, i, j, k].data.numpy()[0]
                        k2 = mat_expected[0][i, j, k].data.numpy()[0]
                        self.assertAlmostEqual(k1, k2, places=4)

        k = 0
        for hiddens_actual, hiddens_expected in zip(all_hiddens_actual, all_hiddens_expected):
            for i in range(hiddens_expected.size()[0]):
                for j in range(hiddens_expected.size()[1]):
                    self.assertAlmostEqual(hiddens_actual[i, j].data.numpy()[0],
                                           hiddens_expected[i, j].data.numpy()[0],
                                           places=4)
            k += 1

        for expd_doc, act_doc in zip(expected.data, actual.data):
            for expd_y, act_y in zip(expd_doc, act_doc):
                self.assertAlmostEqual(expd_y, act_y, places=4)


class Batch:
    def __init__(self, docs, embeddings):
        """ Makes a smaller vocab of only words used in the given docs """
        mini_vocab = Vocab.from_docs(docs, default=0)
        self.docs = [mini_vocab.numberize(doc) for doc in docs]
        local_embeddings = [embeddings[i] for i in mini_vocab.names]
        self.embeddings_matrix = fixed_var(FloatTensor(local_embeddings).t())

    def size(self):
        return len(self.docs)


if __name__ == "__main__":
    unittest.main()
