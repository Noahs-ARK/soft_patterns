#!/usr/bin/env python

import sys
import argparse
import math

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import sigmoid, log_softmax, nll_loss

from data import read_embeddings, read_sentences, read_labels
from mlp import MLP


def score_one_sentence(sentence, embeddings, ws, pi, eta, pattern_length):
    """calculate score for one sentence.
    sentence -- a sequence of indices that correspond to the word embedding matrix"""
    scores = Variable(torch.zeros(len(ws)))
    for pattern_idx, w in enumerate(ws):
        hidden = pi.clone()
        for index in sentence:
            x = embeddings[index]
            delta = compute_delta(x, w, pattern_length)
            hidden = torch.mm(hidden, delta) + pi
            scores[pattern_idx] = scores[pattern_idx] + torch.mm(hidden, eta)

    return scores


def compute_delta(x, w, pattern_length):
    delta = Variable(torch.zeros(pattern_length, pattern_length))

    for i in range(pattern_length):
        for j in range(i, min(i + 2, pattern_length - 1)):
            delta[i, j] = torch.dot(w[i, j], x) - torch.log(torch.norm(w[i, j]))

    return sigmoid(delta)


def train_one_sentence(sentence, embeddings, gold_output, ws, pi, eta, pattern_length, mlp, optimizer):
    """Train one sentence.
    sentence: """
    optimizer.zero_grad()
    scores = score_one_sentence(sentence, embeddings, ws, pi, eta, pattern_length,)

    output = mlp.forward(scores)

    softmax_val = log_softmax(output)
    loss = nll_loss(softmax_val.view(1, 2), Variable(torch.LongTensor([gold_output]), requires_grad=False))
    loss_val = loss.data[0]

    loss.backward()
    optimizer.step()
    return loss_val


# Train model.
# sentences
def train_all(sentences,
              embeddings,
              gold_outputs,
              num_patterns,
              pattern_length,
              word_dim,
              n_iterations,
              mlp_hidden_dim,
              num_classes,
              learning_rate):
    """Train model. sentences -- """
    embeddings = Variable(torch.Tensor(embeddings), requires_grad=False)
    # TODO: why do we need `requires_grad=True`
    ws = [Variable(torch.normal(torch.zeros(pattern_length, pattern_length, word_dim), 1), requires_grad=True)
          for _ in range(num_patterns)]

    # start state distribution
    pi = Variable(torch.zeros(1, pattern_length), requires_grad=False)
    pi[0, 0] = 1
    # end state distribution
    eta = Variable(torch.zeros(pattern_length, 1), requires_grad=False)
    eta[-1, 0] = 1

    mlp = MLP(num_patterns, mlp_hidden_dim, num_classes)

    all_params = list(mlp.parameters()) + ws

    print("# params:", sum(p.nelement() for p in all_params))

    optimizer = torch.optim.Adam(all_params, lr=learning_rate)

    for it in range(n_iterations):
        print("iteration:", it,
              "param_norm:", math.sqrt(sum(p.data.norm() ** 2 for p in all_params)),
              end=" ", flush=True)
        np.random.shuffle(sentences)

        loss = 0.0
        for sentence, gold in zip(sentences, gold_outputs):
            # print(".", end='')
            loss += train_one_sentence(sentence,
                                       embeddings,
                                       gold,
                                       ws,
                                       pi,
                                       eta,
                                       pattern_length,
                                       mlp,
                                       optimizer)
        print("loss:", loss / len(sentences))


def main(args):
    print(args)
    pattern_length = int(args.pattern_length)
    num_patterns = int(args.num_patterns)
    num_iterations = int(args.num_iterations)
    mlp_hidden_dim = int(args.mlp_hidden_dim)

    vocab, embeddings, word_dim = read_embeddings(args.embedding_file)

    train_data = read_sentences(args.td, vocab)
    # dev_data = read_training_data(args.vd, vocab)  # not being used yet

    train_labels = read_labels(args.tl)

    n = args.num_train_instances
    if n is not None:
        train_data = train_data[:n]
        train_labels = train_labels[:n]

    print("training instances:", len(train_data))

    num_classes = len(set(train_labels))
    print("num_classes:", num_classes)

    # dev_labels = read_labels(args.vl)  # not being used yet

    train_all(train_data,
              embeddings,
              train_labels,
              num_patterns,
              pattern_length,
              word_dim,
              num_iterations,
              mlp_hidden_dim,
              num_classes,
              args.learning_rate)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)
    parser.add_argument("-s", "--seed", help="Random seed", default=100)
    parser.add_argument("-i", "--num_iterations", help="Number of iterations", default=10)
    parser.add_argument("-p", "--pattern_length", help="Length of pattern", default=6)
    parser.add_argument("-k", "--num_patterns", help="Number of patterns", default=1)
    parser.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", default=1)
    # TODO: default=None
    parser.add_argument("-n", "--num_train_instances", help="Number of training instances", default=100)
    parser.add_argument("--td", help="Train data file", required=True)
    parser.add_argument("--tl", help="Train labels file", required=True)
    parser.add_argument("--vd", help="Validation data file", required=True)
    parser.add_argument("--vl", help="Validation labels file", required=True)
    parser.add_argument("-l", "--learning_rate", help="Adam Learning rate", default=0.01)

    sys.exit(main(parser.parse_args()))
