#!/usr/bin/env python

import sys
import argparse
import math
import string

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from mlp import MLP


def main(args):
    print(args)
    pattern_length = int(args.pattern_length)
    num_patterns = int(args.num_patterns)
    num_iterations = int(args.num_iterations)
    mlp_hidden_dim = int(args.mlp_hidden_dim)

    vocab, embeddings, word_dim = read_embeddings(args.embedding_file)

    with open(args.td, encoding='utf-8') as ifh:
        train_words = [x.rstrip().split() for x in ifh]
        train_data = [[vocab[w] if w in vocab else 0 for w in sent] for sent in train_words]

    # with open(args.vd, encoding='utf-8') as ifh:
    #     val_words = [x.rstrip().split() for x in ifh]
    #     val_data = [[vocab[w] if w in vocab else 0 for w in sent] for sent in val_words]

    with open(args.tl) as ifh:
        train_labels = [int(x.rstrip()) for x in ifh]

    n = args.num_train_instances
    if n is not None:
        train_data = train_data[:n]
        train_labels = train_labels[:n]

    print("training instances:", len(train_data))

    num_classes = len(set(train_labels))
    print("num_classes:", num_classes)

    # with open(args.vl) as ifh:
    #     val_labels = [int(x.rstrip()) for x in ifh]

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


def read_embeddings(file):
    vocab = dict()
    vocab["*UNK*"] = 0

    vecs = []
    printable = set(string.printable)

    print("Reading", file)
    with open(file, encoding='utf-8') as ifh:
        first_line = ifh.readline()

        e = first_line.rstrip().split()

        # Header line
        if len(e) == 2:
            dim = int(e[1])
            vecs.append(np.zeros(dim))
        else:
            dim = len(e) - 1
            vecs.append(np.zeros(dim))

            vocab[e[0]] = 1
            vecs.append(np.fromstring(" ".join(e[1:]), dtype=float, sep=' '))

        for l in ifh:
            e = l.rstrip().split(" ", 1)
            word = e[0]

            if len(vocab) == 10:
                break
            good = 1
            for i in list(word):
                if i not in printable:
                    good = 0
                    break

            if not good:
                # print(n,"is not good")
                continue

            vocab[e[0]] = len(vocab)
            vecs.append(np.fromstring(e[1], dtype=float, sep=' '))

    print("Done reading", len(vecs), "vectors of dimension", dim)
    return vocab, Variable(torch.Tensor(vecs), requires_grad=False), dim


def score_one_sentence(sentence, embeddings, ws, pi, eta, pattern_length, sigmoid):
    """calculate score for one sentence.
    sentence -- a sequence of indices that correspond to the word embedding matrix"""
    hidden = pi.clone()

    scores = Variable(torch.zeros(len(ws)))
    for pattern_idx, w in enumerate(ws):
        for index in sentence:
            x = embeddings[index]
            delta = compute_delta(x, w, pattern_length, sigmoid)
            hidden = torch.mm(hidden, delta) + pi
            scores[pattern_idx] = scores[pattern_idx] + torch.mm(hidden, eta)

    return scores


def compute_delta(x, w, pattern_length, sigmoid):
    delta = Variable(torch.zeros(pattern_length, pattern_length))

    for i in range(pattern_length):
        for j in range(i, min(i + 2, pattern_length - 1)):
            delta[i, j] = torch.dot(w[i, j], x) - torch.log(torch.norm(w[i, j]))

    return sigmoid(delta)


def train_one_sentence(sentence, embeddings, gold_output, ws, pi, eta, pattern_length, sigmoid, mlp, optimizer):
    """Train one sentence.
    sentence: """
    optimizer.zero_grad()
    scores = score_one_sentence(sentence, embeddings, ws, pi, eta, pattern_length, sigmoid)

    output = mlp.forward(scores)

    softmax = nn.LogSoftmax()
    criterion = nn.NLLLoss()
    softmax_val = softmax(output)
    loss = criterion(softmax_val.view(1, 2), Variable(torch.LongTensor([gold_output]), requires_grad=False))
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
    sigmoid = nn.Sigmoid()

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
                                       sigmoid,
                                       mlp,
                                       optimizer)
        print("loss:", loss / len(sentences))


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
