#!/usr/bin/env python

import sys
import argparse

import numpy as np
from torch import FloatTensor, LongTensor, dot, log, mm, norm, randn, zeros
from torch.autograd import Variable
from torch.functional import stack
from torch.nn import Module, Parameter
from torch.nn.functional import sigmoid, log_softmax, nll_loss
from torch.optim import Adam

from data import read_embeddings, read_docs, read_labels
from mlp import MLP


class SoftPattern(Module):
    """ A single soft pattern """

    def __init__(self,
                 pattern_length,
                 embeddings):
        super(SoftPattern, self).__init__()
        self.pattern_length = pattern_length
        # word vectors (fixed)
        self.embeddings = embeddings
        word_dim = embeddings.size()[1]
        # parameters that determine state transition probabilities based on current word
        self.w = Parameter(randn(pattern_length, pattern_length, word_dim))
        # start state distribution (fixed)
        self.start = Variable(zeros(1, pattern_length), requires_grad=False)
        self.start[0, 0] = 1
        # end state distribution (fixed, for now)
        self.final = Variable(zeros(pattern_length, 1), requires_grad=False)
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
        for i in range(self.pattern_length):
            # only need to look at main diagonal and the two diagonals above it
            for j in range(i, min(i + 2, self.pattern_length - 1)):
                result[i, j] = sigmoid(dot(self.w[i, j], word_vec) - log(norm(self.w[i, j])))

        return result


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
                 embeddings):
        super(SoftPatternClassifier, self).__init__()
        self.embeddings = Variable(FloatTensor(embeddings), requires_grad=False)
        self.patterns = [SoftPattern(pattern_length, self.embeddings) for _ in range(num_patterns)]
        self.mlp = MLP(num_patterns, mlp_hidden_dim, num_classes)
        self.all_params = [p for model in self.patterns + [self.mlp]
                           for p in model.parameters()]
        print("# params:", sum(p.nelement() for p in self.all_params))

    def forward(self, doc):
        scores = stack([p.forward(doc) for p in self.patterns])
        return self.mlp.forward(scores.t())


def train_one_doc(model, doc, gold_output, optimizer):
    """Train on one doc. """
    optimizer.zero_grad()
    output = model.forward(doc)
    loss = nll_loss(
        log_softmax(output).view(1, 2),
        Variable(LongTensor([gold_output]), requires_grad=False)
    )
    loss.backward()
    optimizer.step()
    return loss.data[0]


def train(docs,
          gold_outputs,
          embeddings,
          num_patterns,
          pattern_length,
          num_iterations,
          mlp_hidden_dim,
          num_classes,
          learning_rate):
    """ Train a model on all the given docs """
    model = SoftPatternClassifier(num_patterns, pattern_length, mlp_hidden_dim, num_classes, embeddings)
    optimizer = Adam(model.all_params, lr=learning_rate)

    for it in range(num_iterations):
        print("iteration:", it,
              # "param_norm:", math.sqrt(sum(p.data.norm() ** 2 for p in all_params)),
              end=" ", flush=True)
        np.random.shuffle(docs)

        loss = 0.0
        for sentence, gold in zip(docs, gold_outputs):
            loss += train_one_doc(model, sentence, gold, optimizer)
        print("loss:", loss / len(docs))


def main(args):
    print(args)
    pattern_length = int(args.pattern_length)
    num_patterns = int(args.num_patterns)
    num_iterations = int(args.num_iterations)
    mlp_hidden_dim = int(args.mlp_hidden_dim)

    vocab, embeddings, word_dim = read_embeddings(args.embedding_file)

    train_data = read_docs(args.td, vocab)
    # dev_data = read_training_data(args.vd, vocab)  # not being used yet

    train_labels = read_labels(args.tl)

    # truncate data (to debug faster)
    n = args.num_train_instances
    if n is not None:
        train_data = train_data[:n]
        train_labels = train_labels[:n]

    print("training instances:", len(train_data))

    num_classes = len(set(train_labels))
    print("num_classes:", num_classes)

    # dev_labels = read_labels(args.vl)  # not being used yet

    train(train_data,
          train_labels,
          embeddings,
          num_patterns,
          pattern_length,
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
    parser.add_argument("-k", "--num_patterns", help="Number of patterns", default=2)
    parser.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", default=10)
    # TODO: default=None
    parser.add_argument("-n", "--num_train_instances", help="Number of training instances", default=100)
    parser.add_argument("--td", help="Train data file", required=True)
    parser.add_argument("--tl", help="Train labels file", required=True)
    parser.add_argument("--vd", help="Validation data file", required=True)
    parser.add_argument("--vl", help="Validation labels file", required=True)
    parser.add_argument("-l", "--learning_rate", help="Adam Learning rate", default=1e-3)

    sys.exit(main(parser.parse_args()))
