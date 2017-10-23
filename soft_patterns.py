#!/usr/bin/env python

import sys
import argparse
from time import monotonic

import numpy as np
import os
import torch
from torch import FloatTensor, LongTensor, cat, dot, log, mm, mul, norm, randn, zeros, ones
from torch.autograd import Variable
from torch.functional import stack
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import sigmoid, log_softmax, nll_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import read_embeddings, read_docs, read_labels, vocab_from_text
from mlp import MLP


# Running mode


def fixed_var(tensor, gpu=False):
    if gpu:
        return Variable(tensor, requires_grad=False).cuda()
    else:
        return Variable(tensor, requires_grad=False)


def argmax(output):
    """ only works for kxn tensors """
    _, am = torch.max(output, 1)
    return am


def get_nearest_neighbors(w, embeddings):
    dot_products = mm(w, embeddings[:1000, :].t())

    return argmax(dot_products)


def normalize(data):
    length = data.size()[0]
    for i in range(length):
        data[i] = data[i] / norm(data[i])  # unit length


class Semiring:
    def __init__(self,
                 zero,
                 one,
                 plus,
                 times):
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times


def plus(x, y):
    return x + y


def neg_infinity(*sizes):
    return -100 * ones(*sizes)  # not really -inf, shh


# element-wise plus, times
ProbSemiring = Semiring(zeros, ones, plus, mul)

# element-wise max, plus
MaxPlusSemiring = Semiring(neg_infinity, zeros, torch.max, plus)


class SoftPatternClassifier(Module):
    """
    A text classification model that feeds the document scores from a bunch of
    soft patterns into an MLP
    """

    def __init__(self,
                 pattern_specs,
                 mlp_hidden_dim,
                 num_mlp_layers,
                 num_classes,
                 embeddings,
                 vocab,
                 semiring,
                 gpu=False,
                 dropout=0,
                 legacy=False):
        super(SoftPatternClassifier, self).__init__()
        self.semiring = semiring
        self.vocab = vocab
        self.embeddings = fixed_var(FloatTensor(embeddings), gpu)

        self.dtype = torch.FloatTensor
        if gpu:
            self.embeddings.cuda()
            self.dtype = torch.cuda.FloatTensor

        self.gpu = gpu

        self.total_num_patterns = int(np.sum(list(pattern_specs.values())))

        self.mlp = MLP(self.total_num_patterns, mlp_hidden_dim, num_mlp_layers, num_classes, dropout, legacy)

        self.word_dim = len(embeddings[0])
        self.num_diags = 2  # self-loops and single-forward-steps
        self.pattern_specs = pattern_specs
        self.pattern_lengths = sorted(pattern_specs.keys())
        self.starts = []
        self.ends = []

        diag_data_size = 0

        for i in self.pattern_lengths:
            self.starts.append(diag_data_size)
            diag_data_size += i * self.num_diags * pattern_specs[i]
            self.ends.append(diag_data_size)

        diag_data_size *= self.num_diags
        diag_data = randn(diag_data_size, self.word_dim).type(self.dtype)
        normalize(diag_data)

        bias_data = randn(diag_data_size, 1).type(self.dtype)

        self.dropout = None
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)

        self.diags = Parameter(diag_data)
        self.bias = Parameter(bias_data)
        epsilons = [
            Parameter(randn(self.pattern_specs[pattern_length], (pattern_length - 1)).type(self.dtype))
            for pattern_length in self.pattern_lengths
        ]

        #FIXME
        self.epsilons = epsilons#ModuleList(epsilons)

        # TODO: learned? hyperparameter?
        # self.epsilon_scale = Parameter(randn(1).type(self.dtype))
        # self.self_loop_scale = Parameter(randn(1).type(self.dtype))
        self.epsilon_scale = fixed_var(FloatTensor([.5]).type(self.dtype))
        self.self_loop_scale = fixed_var(FloatTensor([.5]).type(self.dtype))

        #        self.start = fixed_var(zeros(1, pattern_length))
        #        self.start[0, 0] = 1
        # end state distribution (always end in last state)
        #        self.final = fixed_var(zeros(pattern_length, 1))
        #        self.final[-1, 0] = 1

        print("# params:", sum(p.nelement() for p in self.parameters()))

    def visualize_pattern(self, dev_set=None, dev_text=None, n_top_scoring=5):
        embeddings = self.embeddings.data  # torch.transpose(self.embeddings.data, 0, 1)
        nearest_neighbors = get_nearest_neighbors(self.diags.data, embeddings)

        if dev_set is not None:
            # print(dev_set[0])
            scores = self.get_top_scoring_sequences(dev_set)

        start = 0
        for i in range(len(self.pattern_lengths)):
            pattern_length = self.pattern_lengths[i]
            num_patterns = self.pattern_specs[pattern_length]
            print("Visualizing",num_patterns,"patterns of length", pattern_length, self.starts[i], self.ends[i])
            # 1 above main diagonal
            viewed_tensor = self.diags[self.starts[i]:self.ends[i],:].view(num_patterns,
                                            self.num_diags,
                                            pattern_length,
                                            self.word_dim)[:, 1, :-1, :]
            norms = norm(viewed_tensor, 2, 2)
            viewed_biases = self.bias[self.starts[i]:self.ends[i],:].view(num_patterns,
                                           self.num_diags,
                                           pattern_length)[:, 1, :-1]

            # print(norms.size(), viewed_biases.size())
            # for p in range(self.num_patterns):
            #     print("Pattern",p)
            #     for i in range(self.pattern_length-1):
            #         print("\tword",i,"norm",round(norms.data[p,i],3),"bias",round(viewed_biases.data[p,i],3))


            # print(viewed_tensor.size(), (self.num_patterns*(self.pattern_length-1), self.word_dim))
            # reviewed_tensor = viewed_tensor.view(self.num_patterns*(self.pattern_length-1), self.word_dim)

            # print(nearest_neighbors.size(), self.starts[i], self.ends[i], num_patterns, self.num_diags, pattern_length)
            reviewed_nearest_neighbors = \
                nearest_neighbors[self.starts[i]:self.ends[i]].view(num_patterns,
                                       self.num_diags,
                                       pattern_length)[:, 1, :-1]

            if dev_set is not None:
                for p in range(num_patterns):
                    patt_scores = scores[start+p, :, :]
                    # print(scores[0][0])
                    last_n = len(patt_scores) - n_top_scoring
                    sorted_keys = sorted(range(len(patt_scores)), key=lambda i: patt_scores[i][0].data[0])
                    # print(sorted_keys)

                    print("Top scoring",
                          [(" ".join(dev_text[k][int(patt_scores[k][1].data[0]):int(patt_scores[k][2].data[0])]),
                            round(patt_scores[k][0].data[0], 3)) for k in sorted_keys[last_n:]],
                          # "first score", [round(patt_scores[k][3].data[0], 3) for k in sorted_keys[last_n:]],
                          "norms", [round(x, 3) for x in norms.data[p, :]],
                          'biases', [round(x, 3) for x in viewed_biases.data[p, :]],
                          'nearest neighbors', [self.vocab[x] for x in reviewed_nearest_neighbors[p, :]])
                start += num_patterns

    def get_top_scoring_sequences(self, dev_set):
        """
        Get top scoring sequence in doc for this pattern (for interpretation purposes)
        """

        n = 3  # max_score, start_idx, end_idx


        max_scores = Variable(MaxPlusSemiring.zero(self.total_num_patterns, len(dev_set), n))

        zero_paddings = [
            fixed_var(self.semiring.zero(self.pattern_specs[i], 1), self.gpu)
            for i in self.pattern_lengths
        ]

        eps_values = [self.get_eps_value(i) for i in range(len(self.pattern_lengths))]
        self_loop_scale = self.get_self_loop_scale()

        for d, (doc, _) in enumerate(dev_set):
            if (d + 1) % 10 == 0:
                print(".", end="", flush=True)
                # if (d + 1) % 100 == 0:
                #     break  # only visualize first 100 docs

            transition_matrices = self.get_transition_matrices(doc)

            # Todo: to ignore self loops, uncomment the next line
            # for i in range(len(doc) - self.pattern_length + 2):
            # print(self.pattern_lengths, self.pattern_specs)
            for i in range(len(doc)):
                start = 0
                for k in range(len(self.pattern_lengths)):
                    pattern_length = self.pattern_lengths[k]
                    num_patterns = self.pattern_specs[pattern_length]
                    hiddens = Variable(self.semiring.zero(num_patterns, pattern_length).type(self.dtype))

                    # Start state
                    hiddens[:, 0] = self.semiring.one(num_patterns, 1).type(self.dtype)
                    # first_scores = Variable(zeros(num_patterns))

                    # Todo: when we have self loops, uncomment the next line
                    # for j in range(i, len(doc))):
                    # print(d, max_scores.size(), start, num_patterns)
                    for j in range(i, min(i + pattern_length - 1, len(doc))):
                        transition_matrix_val = transition_matrices[j][k]
                        hiddens = self.transition_once(
                            eps_values[k],
                            hiddens,
                            self_loop_scale,
                            transition_matrix_val,
                            zero_paddings[k],
                            zero_paddings[k])

                        # New value for hidden state
                        # hiddens = cat((z1, mul(hiddens[:, :-1], transition_matrices[j][:, 1, :-1])), 1)  # \

                        scores = hiddens[:, -1]

                        # if i == j:
                        #     first_scores = hiddens[:, 1]

                        for p in range(num_patterns):
                            if scores[p].data[0] > max_scores[start+p, d, 0].data[0]:
                                max_scores[start+p, d, 0] = scores[p]
                                max_scores[start+p, d, 1] = i
                                max_scores[start+p, d, 2] = j + 1
                                # max_scores[p, d, 3] = first_scores[p]
                    start += num_patterns

        print()
        return max_scores

    def get_transition_matrices(self, doc):
        # transition matrix for each token in doc
        transition_matrices = []
        for i in range(len(doc)):
            word_index = doc[i]
            x = self.embeddings[word_index].view(self.embeddings.size()[1], 1)

            transition_matrices.append(self.transition_matrix(x))

        return transition_matrices

    def forward(self, doc):
        """
        Calculate score for one document.
        doc -- a sequence of indices that correspond to the word embedding matrix
        """
        transition_matrices = self.get_transition_matrices(doc)

        scores = Variable(self.semiring.zero(self.total_num_patterns).type(self.dtype))


        start = 0
        for i in range(len(self.pattern_lengths)):
            pattern_length = self.pattern_lengths[i]
            num_patterns = self.pattern_specs[pattern_length]
            # hiddens = [ self.start.clone() for _ in range(self.num_patterns)]
            hiddens = Variable(self.semiring.zero(num_patterns, pattern_length).type(self.dtype))

            # Start state
            hiddens[:, 0] = self.semiring.one(num_patterns, 1).type(self.dtype)

            # adding start for each word in the document.
            restart_padding = fixed_var(self.semiring.one(num_patterns, 1), self.gpu)

            zero_padding = fixed_var(self.semiring.zero(num_patterns, 1), self.gpu)

            eps_value = self.get_eps_value(i)
            self_loop_scale = self.get_self_loop_scale()


            for transition_matrix_val in transition_matrices:
                hiddens = self.transition_once(eps_value,
                                               hiddens,
                                               self_loop_scale,
                                               transition_matrix_val[i],
                                               zero_padding,
                                               restart_padding)
                # Score is the final column of hiddens
                scores[start:start+num_patterns] = \
                    self.semiring.plus(scores[start:start+num_patterns], hiddens[:, -1])  # mm(hidden, self.final)  # TODO: change if learning final state
            start += num_patterns

        return self.mlp.forward(stack(scores).t())

        # scores = stack([p.forward(doc) for p in self.patterns])
        # return self.mlp.forward(scores.t())

    def get_self_loop_scale(self):
        return sigmoid(self.self_loop_scale)

    def get_eps_value(self, pattern_length_index):
        return mul(sigmoid(self.epsilon_scale), sigmoid(self.epsilons[pattern_length_index]))

    def transition_once(self,
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
            self.semiring.plus(
                hiddens,
                cat((zero_padding,
                     self.semiring.times(
                         hiddens[:, :-1],
                         eps_value     # doesn't depend on token, just state
                     )), 1))
        # single steps forward (consume a token, move forward one state)
        result = \
            cat((restart_padding,  # <- Adding the start state
                 self.semiring.times(
                     hiddens[:, :-1],
                     transition_matrix_val[:, 1, :-1])
                 ), 1)
        # Adding self loops (consume a token, stay in same state)
        result = \
            self.semiring.plus(
                result,
                mul(self_loop_scale,
                    self.semiring.times(
                        hiddens,
                        transition_matrix_val[:, 0, :]
                    )))
        return result

    def transition_matrix(self, word_vec):
        result = sigmoid(mm(self.diags, word_vec) + self.bias).t()

        if self.gpu:
            result = result.cuda()

        if self.dropout:
            result = self.dropout(result)

        results = []

        for i in range(len(self.pattern_lengths)):
            pattern_length = self.pattern_lengths[i]
            num_patterns = self.pattern_specs[pattern_length]
            start = self.starts[i]
            end = self.ends[i]

            # print(start,end,result.size(), num_patterns, self.num_diags, pattern_length)
            results.append(result[:,start:end].view(
                num_patterns, self.num_diags, pattern_length)
            )


        return results

    def predict(self, doc):
        output = self.forward(doc).data
        return int(argmax(output)[0])


def train_one_doc(model, doc, num_classes, gold_output, optimizer, gpu=False):
    """Train on one doc. """
    optimizer.zero_grad()
    loss = compute_loss(model, doc, num_classes, gold_output, gpu)
    loss.backward()
    optimizer.step()
    return loss.data[0]


def compute_loss(model, doc, num_classes, gold_output, gpu):
    output = model.forward(doc)
    return nll_loss(
        log_softmax(output).view(1, num_classes),
        fixed_var(LongTensor([gold_output]), gpu)
    )


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
          num_classes,
          model_save_dir,
          num_iterations,
          model_file_prefix,
          learning_rate,
          run_scheduler=False,
          gpu=False):
    """ Train a model on all the given docs """
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if run_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    start_time = monotonic()

    for it in range(num_iterations):
        np.random.shuffle(train_data)

        loss = 0.0
        for i, (doc, gold) in enumerate(train_data):
            if i % 100 == 99:
                print(".", end="", flush=True)
            loss += train_one_doc(model, doc, num_classes, gold, optimizer, gpu)

        dev_loss = 0.0
        for i, (doc, gold) in enumerate(dev_data):
            if i % 100 == 99:
                print(".", end="", flush=True)
            dev_loss += compute_loss(model, doc, num_classes, gold, gpu).data[0]

        print("\n")
        finish_iter_time = monotonic()
        train_acc = evaluate_accuracy(model, train_data)
        dev_acc = evaluate_accuracy(model, dev_data)

        # "param_norm:", math.sqrt(sum(p.data.norm() ** 2 for p in all_params)),
        print(
            "iteration: {:>7,} train time: {:>9,.3f}m, eval time: {:>9,.3f}m " \
            "train loss: {:>12,.3f} train_acc: {:>8,.3f}% " \
            "dev loss: {:>12,.3f} dev_acc: {:>8,.3f}%".format(
                # "iteration: {:>7,} time: {:>9,.3f}s loss: {:>12,.3f} dev_acc: {:>8,.3f}%".format(
                it,
                (finish_iter_time - start_time) / 60,
                (monotonic() - finish_iter_time) / 60,
                loss / len(train_data),
                train_acc * 100,
                dev_loss / len(dev_data),
                dev_acc * 100
            )
        )

        if run_scheduler:
            scheduler.step(dev_loss)

        if model_save_dir is not None:
            model_save_file = os.path.join(model_save_dir, "{}_{}.pth".format(model_file_prefix, it))
            torch.save(model.state_dict(), model_save_file)

    return model


def main(args):
    print(args)
    pattern_specs = dict(([int(y) for y in x.split(":")]) for x in args.patterns.split(","))
    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev_vocab = vocab_from_text(args.vd)
    if args.td is not None:
        train_vocab = vocab_from_text(args.td)
        dev_vocab |= train_vocab

    vocab, reverse_vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    dev_input, dev_text = read_docs(args.vd, vocab)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))

    if args.td is not None:
        if args.tl is None:
            print("Both training data (--td) and training labels (--tl) required in training mode")
            return -1

        np.random.shuffle(dev_data)
        num_iterations = args.num_iterations

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
        if args.td is not None:
            train_data = train_data[:n]

        dev_data = dev_data[:n]

    dropout = None if args.td is None else args.dropout
    semiring = MaxPlusSemiring if args.maxplus is None else ProbSemiring

    model = SoftPatternClassifier(pattern_specs,
                                  mlp_hidden_dim,
                                  num_mlp_layers,
                                  num_classes,
                                  embeddings,
                                  reverse_vocab,
                                  semiring,
                                  args.gpu,
                                  dropout,
                                  args.legacy)

    if args.gpu:
        model.cuda()

    model_file_prefix = 'model'
    # Loading model
    if args.input_model is not None:
        state_dict = torch.load(args.input_model)
        model.load_state_dict(state_dict)
        model_file_prefix = 'model_retrained'

    if args.td:
        model_save_dir = args.model_save_dir

        if model_save_dir is not None:
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

        print("Training with", model_file_prefix)
        train(train_data,
              dev_data,
              model,
              num_classes,
              model_save_dir,
              num_iterations,
              model_file_prefix,
              args.learning_rate,
              args.scheduler,
              args.gpu)
    else:
        model.visualize_pattern(dev_data, dev_text)
        # for pattern in model.patterns:
        #     pattern.visualize_pattern(dev_data, dev_text)

        # dev_acc = evaluate_accuracy(model, dev_data)

        # "param_norm:", math.sqrt(sum(p.data.norm() ** 2 for p in all_params)),
        # print("Dev acc:", dev_acc * 100)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)
    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=100)
    parser.add_argument("-i", "--num_iterations", help="Number of iterations", type=int, default=10)
    parser.add_argument("-p", "--patterns",
                        help="Pattern lengths and numbers: a comma separated list of length:number pairs",
                        default="5:50,4:50,3:50,2:50")
    parser.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", type=int, default=10)
    parser.add_argument("-y", "--num_mlp_layers", help="Number of MLP layers", type=int, default=2)
    parser.add_argument("-n", "--num_train_instances", help="Number of training instances", type=int, default=None)
    parser.add_argument("-m", "--model_save_dir", help="where to save the trained model")
    parser.add_argument("-r", "--scheduler", help="Use reduce learning rate on plateau schedule", action='store_true')
    parser.add_argument("-g", "--gpu", help="Use GPU", action='store_true')
    parser.add_argument("-c", "--legacy", help="Load legacy models", action='store_true')
    parser.add_argument("-t", "--dropout", help="Use dropout", type=float, default=0)
    parser.add_argument("--input_model", help="Input model (to run test and not train)")
    parser.add_argument("--td", help="Train data file")
    parser.add_argument("--tl", help="Train labels file")
    parser.add_argument("--vd", help="Validation data file", required=True)
    parser.add_argument("--vl", help="Validation labels file", required=True)
    parser.add_argument("-l", "--learning_rate", help="Adam Learning rate", type=float, default=1e-3)
    parser.add_argument("--maxplus", help="Use max-plus semiring instead of plus-times", action='store_true')

    sys.exit(main(parser.parse_args()))
