#!/usr/bin/env python3
"""
Script to visualize the patterns in a SoftPatterns model based on their
highest-scoring spans in the dev set.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import sys
import torch
import numpy as np
from data import vocab_from_text, read_embeddings, read_docs, read_labels
from soft_patterns import MaxPlusSemiring, LogSpaceMaxTimesSemiring, evaluate_accuracy, SoftPatternClassifier, ProbSemiring, training_arg_parser, \
    soft_pattern_arg_parser

SCORE_IDX = 0
START_IDX_IDX = 1
END_IDX_IDX = 2


# TODO: refactor duplicate code with soft_patterns.py
def main(args):
    print(args)

    pattern_specs = OrderedDict(sorted(([int(y) for y in x.split(":")] for x in args.patterns.split(",")),
                                key=lambda t: t[0]))
    max_pattern_length = max(list(pattern_specs.keys()))
    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab size:", len(dev_vocab))

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev_input, dev_text = read_docs(args.vd, vocab, 0)
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

    model = SoftPatternClassifier(pattern_specs,
                                  mlp_hidden_dim,
                                  num_mlp_layers,
                                  num_classes,
                                  embeddings,
                                  vocab,
                                  semiring,
                                  args.gpu)

    if args.gpu:
        model.to_cuda(model)

    # Loading model
    state_dict = torch.load(args.input_model)
    model.load_state_dict(state_dict)

    test_acc = evaluate_accuracy(model, dev_data, args.batch_size, args.gpu)

    print("Test accuray: {:>8,.3f}%".format(test_acc))

    return 0


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[soft_pattern_arg_parser(), training_arg_parser()])
    sys.exit(main(parser.parse_args()))
