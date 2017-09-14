#!/usr/bin/env python

import sys
import argparse
import string

import torch
import torch.nn
import numpy as np

def main(args):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)
    parser.add_argument("-s", "--seed", help="Random seed", default=100)
    parser.add_argument("-i", "--n_iterations", help="Number of iterations", default=10)
    parser.add_argument("-p", "--pattern_length", help="Length of pattern", default=6)
    parser.add_argument("-k", "--num_of_patterns", help="Number of patterns", default=1)
    parser.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", default=1)
    parser.add_argument("--td", help="Train data file", required=True)
    parser.add_argument("--tl", help="Train labels file", required=True)
    parser.add_argument("--vd", help="Validation data file", required=True)
    parser.add_argument("--vl", help="Validation labels file", required=True)
    parser.add_argument("-l", "--learning_rate", help="Adam Learning rate", default=0.001)

    args = parser.parse_args()
    print(args)

    pattern_length = int(args.pattern_length)
    num_patts = int(args.num_of_patterns)
    n_iterations = int(args.n_iterations)
    mlp_hidden_dim = int(args.mlp_hidden_dim)

    vocab, embeddings, word_dim = read_embeddings(args.embedding_file)

    with open(args.tl) as ifh:
        train_labels = [float(x.rstrip()) for x in ifh]

    num_classes = len(set(train_labels))

    with open(args.vl) as ifh:
        val_labels = [float(x.rstrip()) for x in ifh]

    with open(args.td) as ifh:
        train_words = [x.decode(errors='ignore').rstrip().split() for x in ifh]
        train_data = [[vocab[w] if w in vocab else 0 for w in sent] for sent in train_words]

    with open(args.vd) as ifh:
        val_words = [x.rstrip().split() for x in ifh]
        val_data = [[vocab[w] if w in vocab else 0 for w in sent] for sent in val_words]

    train_all(train_data, embeddings, train_labels, num_patts, pattern_length, word_dim, n_iterations, mlp_hidden_dim, \
              num_classes)

    return 0

def read_embeddings(file):
    dim = -1
    vocab = dict()
    vocab["*UNK*"] = 0

    vecs = []
    printable = set(string.printable)

    print("Reading",file)
    with open(file) as ifh:
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

    print("Done reading",len(vecs),"vectors of dimension",dim)
    # print(vecs)
    return vocab, torch.Tensor(vecs), dim


def score_one_sentence(sentence, embeddings, w, pi, eta, pattern_length, sigmoid):
    """calculate score for one sentence.
    sentence -- a sequence of indices that correspond to the word embedding matrix"""
    hidden = Variable(pi)

    s = 0
    for wp in w:
        for index in sentence:
            x = embeddings[index]
            delta = compute_delta(x, wp, pattern_length, sigmoid)
            hidden += torch.mm(hidden, delta) + pi
            s += torch.dot(hidden, eta)

    return s

def compute_delta(x, w, pattern_length, sigmoid):
    delta = Variable(torch.zeros(pattern_length, pattern_length))

    for i in range(pattern_length):
        for j in range(i, min(i+2, pattern_length-1)):
            delta[i][j] = sigmoid(torch.dot(w[i][j], x) - torch.log(torch.norm(w[i][j])))

    return delta


def train_one_sentence(sentence, embeddings, gold_output, w, pi, eta, pattern_length, sigmoid, mlp, optimizer):
    """Train one sentence.
    sentence: """
    optimizer.zero_grad()
    score = score_one_sentence(sentence, embeddings, w, pi, eta, pattern_length, sigmoid)

    output = mlp.forward(score)

    softmax = nn.LogSoftmax()
    criterion = nn.NLLLoss()
    softmax_val = softmax(output)
    loss = criterion(softmax_val, gold_output)

    loss.backward()
    optimizer.step()

# Train model.
# sentences
def train_all(sentences, embeddings, gold_outputs, num_patterns, pattern_length, word_dim, n_iterations, mlp_hidden_dim, \
              num_classes):
    """Train model. sentences -- """
    sigmoid = nn.Sigmoid()
    mlp = MLP(num_patterns, mlp_hidden_dim, num_classes)

    w = Variable(torch.normal(torch.zeros(num_patterns, pattern_length, pattern_length, word_dim), 1))

    pi = Variable(torch.zeros(pattern_length))
    pi[0] = 1
    eta = Variable(torch.zeros(pattern_length))
    eta[-1] = 1
    optimizer = torch.optim.Adam([w, mlp], lr=0.0001)

    indices = range(sentences.size()[0])

    for i in n_iterations:
        np.random.shuffle(indices)

        for i in indices:
            sentence = sentences[i]
            gold = gold_outputs[i]
            train_one_sentence(sentence, embeddings, gold, w, pi, eta, pattern_length, sigmoid, mlp, optimizer)

if __name__ == '__main__':
    sys.exit(main(sys.argv))