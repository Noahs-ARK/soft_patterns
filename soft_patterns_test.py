#!/usr/bin/env python3
"""
Script to visualize the patterns in a SoftPatterns model based on their
highest-scoring spans in the dev set.
"""
import argparse
from collections import OrderedDict
import sys
import torch
from torch.autograd import Variable
from data import vocab_from_text, read_embeddings, read_docs, read_labels
from soft_patterns import MaxPlusSemiring, evaluate_accuracy, SoftPatternClassifier, ProbSemiring
from util import chunked

SCORE_IDX = 0
START_IDX_IDX = 1
END_IDX_IDX = 2



# TODO: refactor duplicate code with soft_patterns.py
def main(args):
    print(args)

    pattern_specs = OrderedDict([int(y) for y in x.split(":")] for x in args.patterns.split(","))
    max_pattern_length = max(list(pattern_specs.keys()))
    n = args.num_test_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab size:", len(dev_vocab))

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    dev_input, dev_text = read_docs(args.vd, vocab, max_pattern_length/2)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))
    if n is not None:
        dev_data = dev_data[:n]

    num_classes = len(set(dev_labels))
    print("num_classes:", num_classes)

    semiring = MaxPlusSemiring if args.maxplus else ProbSemiring

    epsilon_scale_value = args.epsilon_scale_value if args.epsilon_scale_value is not None else semiring.one([1])
    self_loop_scale_value = args.self_loop_scale_value if args.self_loop_scale_value is not None else semiring.one([1])

    model = SoftPatternClassifier(pattern_specs,
                                  mlp_hidden_dim,
                                  num_mlp_layers,
                                  num_classes,
                                  embeddings,
                                  vocab,
                                  semiring,
                                  epsilon_scale_value,
                                  self_loop_scale_value,
                                  args.gpu,
                                  False)

    if args.gpu:
        model.to_cuda()

    # Loading model
    state_dict = torch.load(args.input_model)
    model.load_state_dict(state_dict)

    test_acc = evaluate_accuracy(model, dev_data, args.batch_size, args.gpu)

    print("Test accuray: {:>8,.3f}%".format(test_acc))

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)
    parser.add_argument("-p", "--patterns",
                        help="Pattern lengths and numbers: a comma separated list of length:number pairs",
                        default="5:50,4:50,3:50,2:50")
    parser.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", type=int, default=10)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=100)
    parser.add_argument("-y", "--num_mlp_layers", help="Number of MLP layers", type=int, default=2)
    parser.add_argument("-n", "--num_test_instances", help="Number of test instances", type=int, default=None)
    parser.add_argument("-g", "--gpu", help="Use GPU", action='store_true')
    parser.add_argument("--input_model", help="Input model (to run test and not train)", required=True)
    parser.add_argument("--vd", help="Validation data file", required=True)
    parser.add_argument("--vl", help="Validation labels file", required=True)
    parser.add_argument("--maxplus",
                        help="Use max-plus semiring instead of plus-times",
                        default=False, action='store_true')
    parser.add_argument("--epsilon_scale_value", help="Value for epsilon scale (default is Semiring.one)", type=float)
    parser.add_argument("--self_loop_scale_value", help="Value for self loop scale (default is Semiring.one)", type=float)

    sys.exit(main(parser.parse_args()))

