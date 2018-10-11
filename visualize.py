#!/usr/bin/env python3
"""
Script to visualize the patterns in a SoftPatterns model based on their
highest-scoring spans in the dev set.
"""
import argparse
from collections import OrderedDict
import sys
from functools import total_ordering
import numpy as np
from soft_patterns import MaxPlusSemiring, fixed_var, Batch, argmax, SoftPatternClassifier, ProbSemiring, \
    LogSpaceMaxTimesSemiring, soft_pattern_arg_parser, general_arg_parser
import torch
from torch.autograd import Variable
from torch.nn import LSTM

from data import vocab_from_text, read_embeddings, read_docs, read_labels
from rnn import Rnn
from util import decreasing_length

SCORE_IDX = 0
START_IDX_IDX = 1
END_IDX_IDX = 2


@total_ordering
class BackPointer:
    def __init__(self,
                 score,
                 previous,
                 transition,
                 start_token_idx,
                 end_token_idx):
        self.score = score
        self.previous = previous
        self.transition = transition
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx

    def __eq__(self, other):
        return self.score == other.score

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return \
            "BackPointer(" \
                "score={}, " \
                "previous={}, " \
                "transition={}, " \
                "start_token_idx={}, " \
                "end_token_idx={}" \
            ")".format(
                self.score,
                self.previous,
                self.transition,
                self.start_token_idx,
                self.end_token_idx
            )

    def display(self, doc_text, extra="", num_padding_tokens=0):
        if self.previous is None:
            return extra  # " ".join("{:<15}".format(s) for s in doc_text[self.start_token_idx:self.end_token_idx])
        if self.transition == "self-loop":
            extra = "SL {:<15}".format(doc_text[self.end_token_idx - 1 - num_padding_tokens]) + extra
            return self.previous.display(doc_text, extra=extra, num_padding_tokens=num_padding_tokens)
        if self.transition == "happy path":
            extra = "HP {:<15}".format(doc_text[self.end_token_idx - 1 - num_padding_tokens]) + extra
            return self.previous.display(doc_text, extra=extra, num_padding_tokens=num_padding_tokens)
        extra = "ep {:<15}".format("") + extra
        return self.previous.display(doc_text, extra=extra, num_padding_tokens=num_padding_tokens)


def get_nearest_neighbors(w, embeddings, k=1000):
    """
    For every transition in every pattern, gets the word with the highest
    score for that transition.
    Only looks at the first `k` words in the vocab (makes sense, assuming
    they're sorted by descending frequency).
    """
    return argmax(torch.mm(w, embeddings[:k, :]))


def visualize_patterns(model,
                       dev_set=None,
                       dev_text=None,
                       k_best=5,
		               max_doc_len=-1,
                       num_padding_tokens=0):
    dev_sorted = decreasing_length(zip(dev_set, dev_text))
    dev_labels = [label for _, label in dev_set]
    dev_set = [doc for doc, _ in dev_sorted]
    dev_text = [text for _, text in dev_sorted]
    num_patterns = model.total_num_patterns
    pattern_length = model.max_pattern_length

    back_pointers = list(get_top_scoring_sequences(model, dev_set, max_doc_len))

    nearest_neighbors = \
        get_nearest_neighbors(
            model.diags.data,
            model.to_cuda(torch.FloatTensor(model.embeddings).t())
        ).view(
            num_patterns,
            model.num_diags,
            pattern_length
        )
    diags = model.diags.view(num_patterns, model.num_diags, pattern_length, model.word_dim).data
    biases = model.bias.view(num_patterns, model.num_diags, pattern_length).data
    self_loop_norms = torch.norm(diags[:, 0, :, :], 2, 2)
    self_loop_neighbs = nearest_neighbors[:, 0, :]
    self_loop_biases = biases[:, 0, :]
    fwd_one_norms = torch.norm(diags[:, 1, :, :], 2, 2)
    fwd_one_biases = biases[:, 1, :]
    fwd_one_neighbs = nearest_neighbors[:, 1, :]
    epsilons = model.get_eps_value().data

    for p in range(num_patterns):
        p_len = model.end_states[p].data[0] + 1
        k_best_doc_idxs = \
            sorted(
                range(len(dev_set)),
                key=lambda doc_idx: back_pointers[doc_idx][p].score,
                reverse=True  # high-scores first
            )[:k_best]

        def span_text(doc_idx):
            back_pointer = back_pointers[doc_idx][p]
            return back_pointer.score, back_pointer.display(dev_text[doc_idx], '#label={}'.format(dev_labels[doc_idx]), num_padding_tokens)

        print("Pattern:", p, "of length", p_len)
        print("Highest scoring spans:")
        for k, d in enumerate(k_best_doc_idxs):
            score, text = span_text(d)
            print("{} {:2.3f}  {}".format(k, score, text.encode('utf-8')))

        def transition_str(norm, neighb, bias):
            return "{:5.2f} * {:<15} + {:5.2f}".format(norm, model.vocab[neighb], bias)

        print("self-loops: ",
              ", ".join(
                  transition_str(norm, neighb, bias)
                  for norm, neighb, bias in zip(self_loop_norms[p, :p_len],
                                                self_loop_neighbs[p, :p_len],
                                                self_loop_biases[p, :p_len])))
        print("fwd 1s:     ",
              ", ".join(
                  transition_str(norm, neighb, bias)
                  for norm, neighb, bias in zip(fwd_one_norms[p, :p_len - 1],
                                                fwd_one_neighbs[p, :p_len - 1],
                                                fwd_one_biases[p, :p_len - 1])))
        print("epsilons:   ",
              ", ".join("{:31.2f}".format(x) for x in epsilons[p, :p_len - 1]))
        print()


def zip_ap_2d(f, a, b):
    return [
        [
            f(x, y) for x, y in zip(xs, ys)
        ]
        for xs, ys in zip(a, b)
    ]


def cat_2d(padding, a):
    return [
        [p] + xs
        for p, xs in zip(padding, a)
    ]


def transition_once_with_trace(model,
                               token_idx,
                               eps_value,
                               back_pointers,
                               transition_matrix_val,
                               restart_padding):
    def times(a, b):
        # wildly inefficient, oh well
        return model.semiring.times(
            torch.FloatTensor([a]),
            torch.FloatTensor([b])
        )[0]

    # Epsilon transitions (don't consume a token, move forward one state)
    # We do this before self-loops and single-steps.
    # We only allow one epsilon transition in a row.
    epsilons = cat_2d(
        restart_padding(token_idx),
        zip_ap_2d(
            lambda bp, e: BackPointer(score=times(bp.score, e),
                                      previous=bp,
                                      transition="epsilon-transition",
                                      start_token_idx=bp.start_token_idx,
                                      end_token_idx=token_idx),
            [xs[:-1] for xs in back_pointers],
            eps_value  # doesn't depend on token, just state
        )
    )

    epsilons = zip_ap_2d(max, back_pointers, epsilons)

    happy_paths = cat_2d(
        restart_padding(token_idx),
        zip_ap_2d(
            lambda bp, t: BackPointer(score=times(bp.score, t),
                                      previous=bp,
                                      transition="happy path",
                                      start_token_idx=bp.start_token_idx,
                                      end_token_idx=token_idx + 1),
            [xs[:-1] for xs in epsilons],
            transition_matrix_val[:, 1, :-1]
        )
    )

    # Adding self loops (consume a token, stay in same state)
    self_loops = zip_ap_2d(
        lambda bp, sl: BackPointer(score=times(bp.score, sl),
                                   previous=bp,
                                   transition="self-loop",
                                   start_token_idx=bp.start_token_idx,
                                   end_token_idx=token_idx + 1),
        epsilons,
        transition_matrix_val[:, 0, :]
    )
    return zip_ap_2d(max, happy_paths, self_loops)


def get_top_scoring_spans_for_doc(model, doc, max_doc_len):
    batch = Batch([doc[0]], model.embeddings, model.to_cuda, 0, max_doc_len)  # single doc
    transition_matrices = model.get_transition_matrices(batch)
    num_patterns = model.total_num_patterns
    end_states = model.end_states.data.view(num_patterns)

    def restart_padding(t):
        return [
            BackPointer(
                score=x,
                previous=None,
                transition=None,
                start_token_idx=t,
                end_token_idx=t
            )
            for x in model.semiring.one(num_patterns)
        ]

    eps_value = model.get_eps_value().data
    hiddens = model.semiring.zero(num_patterns, model.max_pattern_length)
    # set start state activation to 1 for each pattern in each doc
    hiddens[:, 0] = model.semiring.one(num_patterns, 1)
    # convert to back-pointers
    hiddens = \
        [
            [
                BackPointer(
                    score=state_activation,
                    previous=None,
                    transition=None,
                    start_token_idx=0,
                    end_token_idx=0
                )
                for state_activation in pattern
            ]
            for pattern in hiddens
        ]
    # extract end-states
    end_state_back_pointers = [
        bp[end_state]
        for bp, end_state in zip(hiddens, end_states)
    ]
    for token_idx, transition_matrix in enumerate(transition_matrices):
        transition_matrix = transition_matrix[0, :, :, :].data
        hiddens = transition_once_with_trace(model,
                                             token_idx,
                                             eps_value,
                                             hiddens,
                                             transition_matrix,
                                             restart_padding)
        # extract end-states and max with current bests
        end_state_back_pointers = [
            max(best_bp, hidden_bps[end_state])
            for best_bp, hidden_bps, end_state in zip(end_state_back_pointers, hiddens, end_states)
        ]
    return end_state_back_pointers


def get_top_scoring_sequences(model, dev_set, max_doc_len):
    """ Get top scoring sequences for every pattern and doc. """
    for doc_idx, doc in enumerate(dev_set):
        if doc_idx % 100 == 99:
            print(".", end="", flush=True)
        yield get_top_scoring_spans_for_doc(model, doc, max_doc_len)


# TODO: refactor duplicate code with soft_patterns.py
def main(args):
    print(args)

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    pattern_specs = OrderedDict(sorted(([int(y) for y in x.split("-")] for x in args.patterns.split("_")),
                                       key=lambda t: t[0]))

    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab size:", len(dev_vocab))

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    num_padding_tokens = max(list(pattern_specs.keys())) - 1

    dev_input, dev_text = read_docs(args.vd, vocab, num_padding_tokens=num_padding_tokens)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))
    if n is not None:
        dev_data = dev_data[:n]

    num_classes = len(set(dev_labels))
    print("num_classes:", num_classes)

    semiring = \
        MaxPlusSemiring if args.maxplus else (
            LogSpaceMaxTimesSemiring if args.maxtimes else ProbSemiring
        )

    if args.use_rnn:
        rnn = Rnn(word_dim,
                  args.hidden_dim,
                  cell_type=LSTM,
                  gpu=args.gpu)
    else:
        rnn = None

    model = SoftPatternClassifier(pattern_specs, mlp_hidden_dim, num_mlp_layers, num_classes, embeddings, vocab,
                                  semiring, args.bias_scale_param, args.gpu, rnn=rnn, pre_computed_patterns=None,
                                  no_sl=args.no_sl, shared_sl=args.shared_sl, no_eps=args.no_eps,
                                  eps_scale=args.eps_scale, self_loop_scale=args.self_loop_scale)

    if args.gpu:
        state_dict = torch.load(args.input_model)
    else:
        state_dict = torch.load(args.input_model, map_location=lambda storage, loc: storage)

    model.load_state_dict(state_dict)

    if args.gpu:
        model.to_cuda(model)

    visualize_patterns(model, dev_data, dev_text, args.k_best, args.max_doc_len, num_padding_tokens)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[soft_pattern_arg_parser(), general_arg_parser()])
    parser.add_argument("-k", "--k_best", help="Number of nearest neighbor phrases", type=int, default=5)

    sys.exit(main(parser.parse_args()))
