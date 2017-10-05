#!/usr/bin/env python

import sys
import argparse
from time import monotonic

import numpy as np
import os
import torch
from torch import FloatTensor, LongTensor, dot, log, mm, norm, randn, zeros
from torch.autograd import Variable
from torch.functional import stack
from torch.nn import Module, Parameter
from torch.nn.functional import sigmoid, log_softmax, nll_loss
from torch.optim import Adam

from data import read_embeddings, read_docs, read_labels
from mlp import MLP


def fixed_var(tensor):
    return Variable(tensor, requires_grad=False)


def argmax(output):
    """ only works for 1xn tensors """
    _, am = torch.max(output, 1)
    return am[0]


def nearest_neighbor(w, embeddings, vocab):
    return vocab[argmax(mm(w.view(1, w.size()[0]), embeddings))]


class SoftPattern(Module):
    """ A single soft pattern """

    def __init__(self,
                 pattern_length,
                 embeddings,
                 vocab):
        super(SoftPattern, self).__init__()
        self.vocab = vocab
        self.pattern_length = pattern_length
        # word vectors (fixed)
        self.embeddings = embeddings
        word_dim = embeddings.size()[1]
        # parameters that determine state transition probabilities based on current word
        w_data = randn(pattern_length, pattern_length, word_dim)
        for i in range(pattern_length):
            for j in range(pattern_length):
                w_data[i, j] = w_data[i, j] / norm(w_data[i, j])  # unit length
        self.w = Parameter(w_data)
        self.b = Parameter(randn(pattern_length, pattern_length))
        # start state distribution (always start in first state)
        self.start = fixed_var(zeros(1, pattern_length))
        self.start[0, 0] = 1
        # end state distribution (always end in last state)
        self.final = fixed_var(zeros(pattern_length, 1))
        self.final[-1, 0] = 1

    def forward(self, doc):
        """
        Calculate score for one document.
        doc -- a sequence of indices that correspond to the word embedding matrix
        """
        score = Variable(zeros(1))
        hidden = self.start.clone()
        for word_index in doc:
            x = self.embeddings[word_index]
            hidden = mm(hidden, self.transition_matrix(x)) + self.start
            score[0] = score[0] + mm(hidden, self.final)
        return score[0]

    def transition_matrix(self, word_vec):
        result = Variable(zeros(self.pattern_length, self.pattern_length))
        # for i in range(self.pattern_length):
        #     # only need to look at main diagonal and the two diagonals above it
        #     for j in range(i, min(i + 2, self.pattern_length)):
        #         result[i, j] = sigmoid(dot(self.w[i, j], word_vec) - log(norm(self.w[i, j])))
        for i in range(self.pattern_length - 1):
            j = i + 1
            result[i, j] = sigmoid(dot(self.w[i, j], word_vec) + self.b[i, j])

        return result

    def visualize_pattern(self):
        # 1 above main diagonal
        norms = [
            norm(self.w[i, i + 1]).data[0]
            for i in range(self.pattern_length - 1)
        ]
        biases = [
            self.b[i, i + 1].data[0]
            for i in range(self.pattern_length - 1)
        ]
        embeddings = torch.transpose(self.embeddings.data, 0, 1)
        neighbors = [
            nearest_neighbor(self.w[i, i + 1].data, embeddings, self.vocab)
            for i in range(self.pattern_length - 1)
        ]
        print("biases", biases, "norms", norms)
        print("neighbors", neighbors)


class SoftPatternClassifier(Module):
    """
    A text classification model that feeds the document scores from a bunch of
    soft patterns into an MLP
    """

    def __init__(self,
                 num_patterns,
                 pattern_length,
                 mlp_hidden_dim,
                 num_classes,
                 embeddings,
                 vocab):
        super(SoftPatternClassifier, self).__init__()
        self.vocab = vocab
        self.embeddings = fixed_var(FloatTensor(embeddings))
        self.patterns = torch.nn.ModuleList([SoftPattern(pattern_length, self.embeddings, vocab) for _ in range(num_patterns)])
        self.mlp = MLP(num_patterns, mlp_hidden_dim, num_classes)
        print ("# params:", sum(p.nelement() for p in self.parameters()))

    def forward(self, doc):
        scores = stack([p.forward(doc) for p in self.patterns])
        return self.mlp.forward(scores.t())

    def predict(self, doc):
        output = self.forward(doc).data
        return int(argmax(output))


def train_one_doc(model, doc, gold_output, optimizer):
    """Train on one doc. """
    optimizer.zero_grad()
    output = model.forward(doc)
    loss = nll_loss(
        log_softmax(output).view(1, 2),
        fixed_var(LongTensor([gold_output]))
    )
    loss.backward()
    optimizer.step()
    return loss.data[0]


def evaluate_accuracy(model, data):
    n = float(len(data))
    # for doc, gold in data[:10]:
    #     print(gold, model.predict(doc))
    # outputs = [model.forward(doc).data for doc, gold in data[:10]]
    # print(outputs)
    predicted = [model.predict(doc) for doc, gold in data]
    print("num predicted 1s", sum(predicted))
    print("num gold 1s", sum(gold for _, gold in data))
    correct = (1 for pred, (_, gold) in zip(predicted, data) if pred == gold)
    return sum(correct) / n


def train(train_data,
          dev_data,
          model,
          model_save_dir,
          num_iterations,
          learning_rate):
    """ Train a model on all the given docs """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    start_time = monotonic()

    for it in range(num_iterations):
        np.random.shuffle(train_data)

        for pattern in model.patterns:
            pattern.visualize_pattern()

        loss = 0.0
        for i, (doc, gold) in enumerate(train_data):
            if i % 10 == 9:
                print(".", end="", flush=True)
            loss += train_one_doc(model, doc, gold, optimizer)

        train_acc = evaluate_accuracy(model, train_data)
        dev_acc = evaluate_accuracy(model, dev_data)
        # "param_norm:", math.sqrt(sum(p.data.norm() ** 2 for p in all_params)),
        print(
            "iteration: {:>7,} time: {:>9,.3f}s loss: {:>12,.3f} train_acc: {:>8,.3f}% dev_acc: {:>8,.3f}%".format(
            # "iteration: {:>7,} time: {:>9,.3f}s loss: {:>12,.3f} dev_acc: {:>8,.3f}%".format(
                it,
                monotonic() - start_time,
                loss / len(train_data),
                train_acc * 100,
                dev_acc * 100
            )
        )
        model_save_file = os.path.join(model_save_dir, "model_{}.pth".format(it))
        torch.save(model.state_dict(), model_save_file)

    return model


def main(args):
    print(args)
    pattern_length = int(args.pattern_length)
    num_patterns = int(args.num_patterns)
    num_iterations = int(args.num_iterations)
    mlp_hidden_dim = int(args.mlp_hidden_dim)

    vocab, reverse_vocab, embeddings, word_dim =\
        read_embeddings(args.embedding_file)

    train_input = read_docs(args.td, vocab)
    train_labels = read_labels(args.tl)

    print("training instances:", len(train_input))

    num_classes = len(set(train_labels))
    print("num_classes:", num_classes)

    # truncate data (to debug faster)
    n = args.num_train_instances
    train_data = list(zip(train_input, train_labels))
    np.random.shuffle(train_data)

    dev_input = read_docs(args.vd, vocab)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))
    np.random.shuffle(dev_data)

    if n is not None:
        n = int(n)
        train_data = train_data[:n]
        dev_data = dev_data[:n]

    model = SoftPatternClassifier(num_patterns,
                                  pattern_length,
                                  mlp_hidden_dim,
                                  num_classes,
                                  embeddings,
                                  reverse_vocab)

    model_save_dir = args.model_save_dir

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train(train_data,
          dev_data,
          model,
          model_save_dir,
          num_iterations,
          args.learning_rate)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)
    parser.add_argument("-s", "--seed", help="Random seed", default=100)
    parser.add_argument("-i", "--num_iterations", help="Number of iterations", default=10)
    parser.add_argument("-p", "--pattern_length", help="Length of pattern", default=2)
    parser.add_argument("-k", "--num_patterns", help="Number of patterns", type=int, default=2)
    parser.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", default=10)
    parser.add_argument("-n", "--num_train_instances", help="Number of training instances", default=None)
    parser.add_argument("-m", "--model_save_dir", help="where to save the trained model", required=True)
    parser.add_argument("--td", help="Train data file", required=True)
    parser.add_argument("--tl", help="Train labels file", required=True)
    parser.add_argument("--vd", help="Validation data file", required=True)
    parser.add_argument("--vl", help="Validation labels file", required=True)
    parser.add_argument("-l", "--learning_rate", help="Adam Learning rate", type=float, default=1e-3)

    sys.exit(main(parser.parse_args()))
