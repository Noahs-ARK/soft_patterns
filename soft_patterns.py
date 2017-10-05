#!/usr/bin/env python

import sys
import argparse
from time import monotonic

import numpy as np
import os
import torch
from torch import FloatTensor, LongTensor, cat, dot, log, mm, mul, norm, randn, zeros
from torch.autograd import Variable
from torch.functional import stack
from torch.nn import Module, Parameter
from torch.nn.functional import sigmoid, log_softmax, nll_loss
from torch.optim import Adam

from data import read_embeddings, read_docs, read_labels
from mlp import MLP

# Running mode


def fixed_var(tensor):
    return Variable(tensor, requires_grad=False)


def argmax(output):
    """ only works for 1xn tensors """
    _, am = torch.max(output, 1)
    return am[0]


def nearest_neighbor(w, embeddings, vocab):
    return vocab[argmax(mm(w.view(1, w.size()[0]), embeddings))]


def normalize(data):
    length = data.size()[0]
    for i in range(length):
        data[i] = data[i] / norm(data[i])  # unit length


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
        word_dim = embeddings.size()[0]
        self.num_diags = 3
        # parameters that determine state transition probabilities based on current word
        diag_data = randn(self.num_diags, word_dim, pattern_length)
        # self_loop_data = randn(word_dim, pattern_length)
        # one_forward_data = randn(word_dim, pattern_length - 1)
        # two_forward_data = randn(word_dim, pattern_length - 2)
        # w_data = randn(pattern_length, pattern_length, word_dim)
        # normalize(self_loop_data)
        for diag in diag_data:
            normalize(diag)
        self.diags = Parameter(diag_data)
        # normalize(two_forward_data)
        # self.w = Parameter(w_data)
        # self.self_loop = Parameter(self_loop_data)
        # self.one_forward = Parameter(one_forward_data)

        # self.two_forward = Parameter(two_forward_data)
        # self.b = Parameter(randn(pattern_length, pattern_length))
        # self.self_loop_bias = Parameter(randn(1, pattern_length))
        # self.one_forward_bias = Parameter(randn(1, pattern_length - 1))
        self.bias = Parameter(randn(self.num_diags, pattern_length))
        # self.two_forward_bias = Parameter(randn(1, pattern_length - 2))
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
        z1 = Variable(zeros(1, 1))
        # z2 = Variable(zeros(1, 2))
        for word_index in doc:
            x = self.embeddings[:, word_index].view(1, self.embeddings.size()[0])
            self_loop_result, one_forward_result, two_forward_result = \
                self.transition_matrix(x)
            # mul(hidden, self_loop_result) + \
            # print("z1", z1.size(),
            #       "hidden", hidden.size(),
            #       "one_forward_result", one_forward_result.size(),
            #       # hidden[:, :-1].size(), one_forward_result.size(),
            #       "mul", mul(hidden[:, :-1], one_forward_result).size())
            hidden = self.start + \
                         cat((z1, mul(hidden[:, :-1], one_forward_result)), 1)
                         # cat((z2, mul(hidden[:, :-2], two_forward_result)), 1)
            score[0] = score[0] + hidden[0, -1]  # mm(hidden, self.final)  # TODO: change if learning final state
        return score[0]

    # FIXME: doesn't work with new vectorized transition matrices
    def get_top_scoring_sequence(self, doc):
        """
        Get top scoring sequence in doc for this pattern (for intepretation purposes)
        """
        max_score = Variable(zeros(1))
        max_start = -1
        max_end = -1

        for i in range(len(doc)-1):
            word_index = doc[i]
            x = self.embeddings[:, word_index]
            score = Variable(zeros(1))
            hidden = mm(self.start.clone(), self.transition_matrix(x))
            score[0] = score[0] + mm(hidden, self.final)

            if score[0].data[0] > max_score[0].data[0]:
                max_score[0] = score[0]
                max_start = i
                max_end = i+1

            for j in range(i+1, len(doc)):
                word_index2 = doc[j]
                y = self.embeddings[:, word_index2]
                hidden = mm(hidden, self.transition_matrix(y))
                score[0] = score[0] + mm(hidden, self.final)

                if score[0].data[0] > max_score[0].data[0]:
                    max_score[0] = score[0]
                    max_start = i
                    max_end = j+1

        # print(max_score[0].data[0], max_start, max_end)
        return max_score[0], max_start, max_end

    def transition_matrix(self, word_vec):
        # result = Variable(zeros(self.pattern_length, self.pattern_length))
        # for i in range(self.pattern_length):
        #     # only need to look at main diagonal and the two diagonals above it
        #     for j in range(i, min(i + 2, self.pattern_length)):
        #         result[i, j] = sigmoid(dot(self.w[i, j], word_vec) - log(norm(self.w[i, j])))
        # for i in range(self.pattern_length - 1):
        #     j = i + 1
        #     result[i, j] = sigmoid(dot(self.w[i, j], word_vec) + self.b[i, j])
        # self_loop_result = sigmoid(mm(word_vec, self.self_loop) + self.self_loop_bias)
        # print(self.one_forward.size(), word_vec.size(), self.one_forward_bias.size())
        # print(mm(word_vec.view(1, word_vec.size()[0]), self.one_forward).size())
        # one_forward_result = sigmoid(mm(word_vec, self.one_forward) + self.one_forward_bias)
        # two_forward_result = sigmoid(mm(word_vec, self.two_forward) + self.two_forward_bias)

        result = [
            sigmoid(mm(word_vec, self.diag) + self.bias[i])
            for i, diag in enumerate(self.diags)
        ]

        return result

    # FIXME: doesn't work with new vectorized transition matrices
    def visualize_pattern(self, dev_set = None, dev_text = None, n_top_scoring = 5):
        # 1 above main diagonal
        norms = [
            norm(self.w[i, i + 1]).data[0]
            for i in range(self.pattern_length - 1)
        ]
        biases = [
            self.b[i, i + 1].data[0]
            for i in range(self.pattern_length - 1)
        ]
        embeddings = self.embeddings.data  # torch.transpose(self.embeddings.data, 0, 1)
        neighbors = [
            nearest_neighbor(self.w[i, i + 1].data, embeddings, self.vocab)
            for i in range(self.pattern_length - 1)
        ]
        print("biases", biases, "norms", norms)
        print("neighbors", neighbors)

        if dev_set is not None:
            # print(dev_set[0])
            scores = [
                self.get_top_scoring_sequence(doc) for doc,_ in dev_set
            ]

            # print(scores[0][0])
            last_n = len(scores)-n_top_scoring
            sorted_keys = sorted(range(len(scores)), key=lambda i: scores[i][0].data[0])
            # print(sorted_keys)


            print("Top scoring", [(" ".join(dev_text[k][scores[k][1]:scores[k][2]]),
                                   round(scores[k][0].data[0], 3)) for k in sorted_keys[last_n:]])


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
        self.embeddings = fixed_var(FloatTensor(embeddings).t())
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

        # for pattern in model.patterns:
        #     pattern.visualize_pattern()

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

        if model_save_dir is not None:
            model_save_file = os.path.join(model_save_dir, "model_{}.pth".format(it))
            torch.save(model.state_dict(), model_save_file)

    return model


def main(args):
    print(args)
    pattern_length = args.pattern_length
    num_patterns = args.num_patterns
    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim

    vocab, reverse_vocab, embeddings, word_dim =\
        read_embeddings(args.embedding_file)

    dev_input, dev_text = read_docs(args.vd, vocab)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))

    if args.input_model is None:
        np.random.shuffle(dev_data)
        num_iterations = args.num_iterations
        if args.td is None or args.tl is None:
            print("Both training data (--td) and training labels (--tl) required in training mode")
            return -1

        train_input, _ = read_docs(args.td, vocab)
        train_labels = read_labels(args.tl)

        print("training instances:", len(train_input))

        num_classes = len(set(train_labels))

        # truncate data (to debug faster)
        train_data = list(zip(train_input, train_labels))
        np.random.shuffle(train_data)
    else:
        num_classes = len(set(dev_labels))

    print("num_classes:", num_classes)

    if n is not None:
        if args.input_model is None:
            train_data = train_data[:n]
        dev_data = dev_data[:n]

    model = SoftPatternClassifier(num_patterns,
                                  pattern_length,
                                  mlp_hidden_dim,
                                  num_classes,
                                  embeddings,
                                  reverse_vocab)

    if args.input_model is None:
        model_save_dir = args.model_save_dir

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        train(train_data,
              dev_data,
              model,
              model_save_dir,
              num_iterations,
              args.learning_rate)
    else:
        state_dict = torch.load(args.input_model)
        model.load_state_dict(state_dict)

        # for pattern in model.patterns:
        #     pattern.visualize_pattern(dev_data, dev_text)

        dev_acc = evaluate_accuracy(model, dev_data)

        # "param_norm:", math.sqrt(sum(p.data.norm() ** 2 for p in all_params)),
        print("Dev acc:", dev_acc * 100)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)
    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=100)
    parser.add_argument("-i", "--num_iterations", help="Number of iterations", type=int, default=10)
    parser.add_argument("-p", "--pattern_length", help="Length of pattern", type=int, default=2)
    parser.add_argument("-k", "--num_patterns", help="Number of patterns", type=int, default=2)
    parser.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", type=int, default=10)
    parser.add_argument("-n", "--num_train_instances", help="Number of training instances", type=int, default=None)
    parser.add_argument("-m", "--model_save_dir", help="where to save the trained model")
    parser.add_argument("--input_model", help="Input model (to run test and not train)")
    parser.add_argument("--td", help="Train data file")
    parser.add_argument("--tl", help="Train labels file")
    parser.add_argument("--vd", help="Validation data file", required=True)
    parser.add_argument("--vl", help="Validation labels file", required=True)
    parser.add_argument("-l", "--learning_rate", help="Adam Learning rate", type=float, default=1e-3)

    sys.exit(main(parser.parse_args()))
