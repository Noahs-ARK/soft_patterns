#!/usr/bin/env python3 -u
"""
A text classification model that feeds the document scores from a bunch of
soft patterns into an MLP.
"""

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from time import monotonic
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.autograd import Variable
from torch.nn import Module, Parameter, NLLLoss, LSTM
from torch.nn.functional import sigmoid, log_softmax
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rnn import lstm_arg_parser, Rnn
from data import read_embeddings, read_docs, read_labels, vocab_from_text, Vocab, UNK_IDX, START_TOKEN_IDX, \
    END_TOKEN_IDX
from mlp import MLP, mlp_arg_parser
from util import shuffled_chunked_sorted, identity, chunked_sorted, to_cuda, right_pad

CW_TOKEN = "CW"
EPSILON = 1e-10


def fixed_var(tensor):
    return Variable(tensor, requires_grad=False)


def argmax(output):
    """ only works for kxn tensors """
    _, am = torch.max(output, 1)
    return am


def normalize(data):
    length = data.size()[0]
    for i in range(length):
        data[i] = data[i] / norm(data[i])  # unit length


class Semiring:
    def __init__(self,
                 zero,
                 one,
                 plus,
                 times,
                 from_float,
                 to_float):
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times
        self.from_float = from_float
        self.to_float = to_float


def neg_infinity(*sizes):
    return -100 * ones(*sizes)  # not really -inf, shh


# element-wise plus, times
ProbSemiring = \
    Semiring(
        zeros,
        ones,
        torch.add,
        torch.mul,
        sigmoid,
        identity
    )

# element-wise max, plus
MaxPlusSemiring = \
    Semiring(
        neg_infinity,
        zeros,
        torch.max,
        torch.add,
        identity,
        identity
    )
# element-wise max, times. in log-space
LogSpaceMaxTimesSemiring = \
    Semiring(
        neg_infinity,
        zeros,
        torch.max,
        torch.add,
        lambda x: torch.log(torch.sigmoid(x)),
        torch.exp
    )

SHARED_SL_PARAM_PER_STATE_PER_PATTERN = 1
SHARED_SL_SINGLE_PARAM = 2

### Adapted from AllenNLP
def enable_gradient_clipping(model, clip) -> None:
    if clip is not None and clip > 0:
        # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
        # pylint: disable=invalid-unary-operand-type
        clip_function = lambda grad: grad.clamp(-clip, clip)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(clip_function)



class Batch:
    """
    A batch of documents.
    Handles truncating documents to `max_len`, looking up word embeddings,
    and padding so that all docs in the batch have the same length.
    Makes a smaller vocab and embeddings matrix, only including words that are in the batch.
    """
    def __init__(self, docs, embeddings, cuda, word_dropout=0, max_len=-1):
        # print(docs)
        mini_vocab = Vocab.from_docs(docs, default=UNK_IDX, start=START_TOKEN_IDX, end=END_TOKEN_IDX)
        # Limit maximum document length (for efficiency reasons).
        if max_len != -1:
            docs = [doc[:max_len] for doc in docs]
        doc_lens = [len(doc) for doc in docs]
        self.doc_lens = cuda(torch.LongTensor(doc_lens))
        self.max_doc_len = max(doc_lens)
        if word_dropout:
            # for each token, with probability `word_dropout`, replace word index with UNK_IDX.
            docs = [
                [UNK_IDX if np.random.rand() < word_dropout else x for x in doc]
                for doc in docs
            ]
        # pad docs so they all have the same length.
        # we pad with UNK, whose embedding is 0, so it doesn't mess up sums or averages.
        docs = [right_pad(mini_vocab.numberize(doc), self.max_doc_len, UNK_IDX) for doc in docs]
        self.docs = [cuda(fixed_var(torch.LongTensor(doc))) for doc in docs]
        local_embeddings = [embeddings[i] for i in mini_vocab.names]
        self.embeddings_matrix = cuda(fixed_var(FloatTensor(local_embeddings).t()))

    def size(self):
        return len(self.docs)


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
                 bias_scale_param,
                 gpu=False,
                 rnn=None,
                 pre_computed_patterns=None,
                 no_sl=False,
                 shared_sl=False,
                 no_eps=False,
                 eps_scale=None,
                 self_loop_scale=None):
        super(SoftPatternClassifier, self).__init__()
        self.semiring = semiring
        self.vocab = vocab
        self.embeddings = embeddings

        self.to_cuda = to_cuda(gpu)

        self.total_num_patterns = sum(pattern_specs.values())
        print(self.total_num_patterns, pattern_specs)
        self.rnn = rnn
        self.mlp = MLP(self.total_num_patterns, mlp_hidden_dim, num_mlp_layers, num_classes)

        if self.rnn is None:
            self.word_dim = len(embeddings[0])
        else:
            self.word_dim = self.rnn.num_directions * self.rnn.hidden_dim
        self.num_diags = 1  # self-loops and single-forward-steps
        self.no_sl = no_sl
        self.shared_sl = shared_sl

        self.pattern_specs = pattern_specs
        self.max_pattern_length = max(list(pattern_specs.keys()))

        self.no_eps = no_eps
        self.bias_scale_param = bias_scale_param

        # Shared parameters between main path and self loop.
        # 1 -- one parameter per state per pattern
        # 2 -- a single global parameter
        if self.shared_sl > 0:
            if self.shared_sl == SHARED_SL_PARAM_PER_STATE_PER_PATTERN:
                shared_sl_data = randn(self.total_num_patterns, self.max_pattern_length)
            elif self.shared_sl == SHARED_SL_SINGLE_PARAM:
                shared_sl_data = randn(1)

            self.self_loop_scale = Parameter(shared_sl_data)
        elif not self.no_sl:
            if self_loop_scale is not None:
                self.self_loop_scale = self.semiring.from_float(self.to_cuda(fixed_var(FloatTensor([self_loop_scale]))))
            else:
                self.self_loop_scale = self.to_cuda(fixed_var(semiring.one(1)))
            self.num_diags = 2

        # end state index for each pattern
        end_states = [
            [end]
            for pattern_len, num_patterns in self.pattern_specs.items()
            for end in num_patterns * [pattern_len - 1]
        ]

        self.end_states = self.to_cuda(fixed_var(LongTensor(end_states)))

        diag_data_size = self.total_num_patterns * self.num_diags * self.max_pattern_length
        diag_data = randn(diag_data_size, self.word_dim)
        bias_data = randn(diag_data_size, 1)

        normalize(diag_data)

        if pre_computed_patterns is not None:
            diag_data, bias_data = self.load_pre_computed_patterns(pre_computed_patterns, diag_data, bias_data, pattern_specs)

        self.diags = Parameter(diag_data)

        # Bias term
        self.bias = Parameter(bias_data)

        if not self.no_eps:
            self.epsilon = Parameter(randn(self.total_num_patterns, self.max_pattern_length - 1))

        # TODO: learned? hyperparameter?
            # since these are currently fixed to `semiring.one`, they are not doing anything.
            if eps_scale is not None:
                self.epsilon_scale = self.semiring.from_float(self.to_cuda(fixed_var(FloatTensor([eps_scale]))))
            else:
                self.epsilon_scale = self.to_cuda(fixed_var(semiring.one(1)))

        print("# params:", sum(p.nelement() for p in self.parameters()))

    def get_transition_matrices(self, batch, dropout=None):
        b = batch.size()
        n = batch.max_doc_len
        if self.rnn is None:
            transition_scores = \
                self.semiring.from_float(mm(self.diags, batch.embeddings_matrix) + self.bias_scale_param * self.bias).t()
            if dropout is not None and dropout:
                transition_scores = dropout(transition_scores)
            batched_transition_scores = [
                torch.index_select(transition_scores, 0, doc) for doc in batch.docs
            ]
            batched_transition_scores = torch.cat(batched_transition_scores).view(
                b, n, self.total_num_patterns, self.num_diags, self.max_pattern_length)

        else:
            # run an RNN to get the word vectors to input into our soft-patterns
            outs = self.rnn.forward(batch, dropout=dropout)
            padded, _ = pad_packed_sequence(outs, batch_first=True)
            padded = padded.contiguous().view(b * n, self.word_dim).t()

            if dropout is not None and dropout:
                padded = dropout(padded)

            batched_transition_scores = \
                self.semiring.from_float(mm(self.diags, padded) + self.bias_scale_param * self.bias).t()

            if dropout is not None and dropout:
                batched_transition_scores = dropout(batched_transition_scores)

            batched_transition_scores = \
                batched_transition_scores.contiguous().view(
                    b,
                    n,
                    self.total_num_patterns,
                    self.num_diags,
                    self.max_pattern_length
                )
        # transition matrix for each token idx
        transition_matrices = [
            batched_transition_scores[:, word_index, :, :, :]
            for word_index in range(n)
        ]
        return transition_matrices

    def load_pre_computed_patterns(self, pre_computed_patterns, diag_data, bias_data, pattern_spec):
        """Loading a set of pre-coputed patterns into diagonal and bias arrays"""
        pattern_indices = dict((p,0) for p in pattern_spec)

        # First,view diag_data and bias_data as 4/3d tensors
        diag_data_size = diag_data.size()[0]
        diag_data = diag_data.view(self.total_num_patterns, self.num_diags, self.max_pattern_length, self.word_dim)
        bias_data = bias_data.view(self.total_num_patterns, self.num_diags, self.max_pattern_length)

        n = 0

        # Pattern indices: which patterns are we loading?
        # the pattern index from which we start loading each pattern length.
        for (i, patt_len) in enumerate(pattern_spec.keys()):
            pattern_indices[patt_len] = n
            n += pattern_spec[patt_len]

        # Loading all pre-computed patterns
        for p in pre_computed_patterns:
            patt_len = len(p) + 1

            # Getting pattern index in diagonal data
            index = pattern_indices[patt_len]

            # Loading diagonal and bias for p
            diag, bias = self.load_pattern(p)

            # Updating diagonal and bias
            diag_data[index, 1, :(patt_len-1), :] = diag
            bias_data[index, 1, :(patt_len-1)] = bias

            # Updating pattern_indices
            pattern_indices[patt_len] += 1

        return diag_data.view(diag_data_size, self.word_dim), bias_data.view(diag_data_size, 1)

    def load_pattern(self, patt):
        """Loading diagonal and bias of one pattern"""
        diag = EPSILON * torch.randn(len(patt), self.word_dim)
        bias = torch.zeros(len(patt))

        factor = 10

        # Traversing elements of pattern.
        for (i, element) in enumerate(patt):
            # CW: high bias (we don't care about the identity of the token
            if element == CW_TOKEN:
                bias[i] = factor
            else:
                # Concrete word: we do care about the token (low bias).
                bias[i] = -factor

                # If we have a word vector for this element, the diagonal value if this vector
                if element in self.vocab:
                    diag[i] = FloatTensor(factor*self.embeddings[self.vocab.index[element]])

        return diag, bias

    def forward(self, batch, debug=0, dropout=None):
        """ Calculate scores for one batch of documents. """
        time1 = monotonic()
        transition_matrices = self.get_transition_matrices(batch, dropout)
        time2 = monotonic()

        self_loop_scale = None

        if self.shared_sl:
            self_loop_scale = self.semiring.from_float(self.self_loop_scale)
        elif not self.no_sl:
            self_loop_scale = self.self_loop_scale

        batch_size = batch.size()
        num_patterns = self.total_num_patterns
        scores = self.to_cuda(fixed_var(self.semiring.zero(batch_size, num_patterns)))

        # to add start state for each word in the document.
        restart_padding = self.to_cuda(fixed_var(self.semiring.one(batch_size, num_patterns, 1)))

        zero_padding = self.to_cuda(fixed_var(self.semiring.zero(batch_size, num_patterns, 1)))

        eps_value = self.get_eps_value()

        batch_end_state_idxs = self.end_states.expand(batch_size, num_patterns, 1)
        hiddens = self.to_cuda(Variable(self.semiring.zero(batch_size,
                                                           num_patterns,
                                                           self.max_pattern_length)))
        # set start state (0) to 1 for each pattern in each doc
        hiddens[:, :, 0] = self.to_cuda(self.semiring.one(batch_size, num_patterns, 1))

        if debug % 4 == 3:
            all_hiddens = [hiddens[0, :, :]]
        for i, transition_matrix in enumerate(transition_matrices):
            hiddens = self.transition_once(eps_value,
                                           hiddens,
                                           transition_matrix,
                                           zero_padding,
                                           restart_padding,
                                           self_loop_scale)
            if debug % 4 == 3:
                all_hiddens.append(hiddens[0, :, :])

            # Look at the end state for each pattern, and "add" it into score
            end_state_vals = torch.gather(hiddens, 2, batch_end_state_idxs).view(batch_size, num_patterns)
            # but only update score when we're not already past the end of the doc
            active_doc_idxs = torch.nonzero(torch.gt(batch.doc_lens, i)).squeeze()
            scores[active_doc_idxs] = \
                self.semiring.plus(
                    scores[active_doc_idxs],
                    end_state_vals[active_doc_idxs]
                )

        if debug:
            time3 = monotonic()
            print("MM: {}, other: {}".format(round(time2 - time1, 3), round(time3 - time2, 3)))

        scores = self.semiring.to_float(scores)

        if debug % 4 == 3:
            return self.mlp.forward(scores), transition_matrices, all_hiddens
        elif debug % 4 == 1:
            return self.mlp.forward(scores), scores
        else:
            return self.mlp.forward(scores)

    def get_eps_value(self):
        return None if self.no_eps else self.semiring.times(
            self.epsilon_scale,
            self.semiring.from_float(self.epsilon)
        )

    def transition_once(self,
                        eps_value,
                        hiddens,
                        transition_matrix_val,
                        zero_padding,
                        restart_padding,
                        self_loop_scale):
        # Adding epsilon transitions (don't consume a token, move forward one state)
        # We do this before self-loops and single-steps.
        # We only allow zero or one epsilon transition in a row.
        if self.no_eps:
            after_epsilons = hiddens
        else:
            after_epsilons = \
                self.semiring.plus(
                    hiddens,
                    cat((zero_padding,
                         self.semiring.times(
                             hiddens[:, :, :-1],
                             eps_value  # doesn't depend on token, just state
                         )), 2)
                )

        after_main_paths = \
            cat((restart_padding,  # <- Adding the start state
                 self.semiring.times(
                     after_epsilons[:, :, :-1],
                     transition_matrix_val[:, :, -1, :-1])
                 ), 2)

        if self.no_sl:
            return after_main_paths
        else:
            self_loop_scale = self_loop_scale.expand(transition_matrix_val[:, :, 0, :].size()) \
                if self.shared_sl == SHARED_SL_PARAM_PER_STATE_PER_PATTERN else self_loop_scale

            # Adding self loops (consume a token, stay in same state)
            after_self_loops = self.semiring.times(
                self_loop_scale,
                self.semiring.times(
                    after_epsilons,
                    transition_matrix_val[:, :, 0, :]
                )
            )
            # either happy or self-loop, not both
            return self.semiring.plus(after_main_paths, after_self_loops)

    def predict(self, batch, debug=0):
        output = self.forward(batch, debug).data
        return [int(x) for x in argmax(output)]


def train_batch(model, batch, num_classes, gold_output, optimizer, loss_function, gpu=False, debug=0, dropout=None):
    """Train on one doc. """
    optimizer.zero_grad()
    time0 = monotonic()
    loss = compute_loss(model, batch, num_classes, gold_output, loss_function, gpu, debug, dropout)
    time1 = monotonic()
    loss.backward()
    time2 = monotonic()

    optimizer.step()
    if debug:
        time3 = monotonic()
        print("Time in loss: {}, time in backward: {}, time in step: {}".format(round(time1 - time0, 3),
                                                                                round(time2 - time1, 3),
                                                                                round(time3 - time2, 3)))
    return loss.data


def compute_loss(model, batch, num_classes, gold_output, loss_function, gpu, debug=0, dropout=None):
    time1 = monotonic()
    output = model.forward(batch, debug, dropout)

    if debug:
        time2 = monotonic()
        print("Forward total in loss: {}".format(round(time2 - time1, 3)))

    return loss_function(
        log_softmax(output).view(batch.size(), num_classes),
        to_cuda(gpu)(fixed_var(LongTensor(gold_output)))
    )


def evaluate_accuracy(model, data, batch_size, gpu, debug=0):
    n = float(len(data))
    correct = 0
    num_1s = 0
    for batch in chunked_sorted(data, batch_size):
        batch_obj = Batch([x for x, y in batch], model.embeddings, to_cuda(gpu))
        gold = [y for x, y in batch]
        predicted = model.predict(batch_obj, debug)
        num_1s += predicted.count(1)
        correct += sum(1 for pred, gold in zip(predicted, gold) if pred == gold)

    print("num predicted 1s:", num_1s)
    print("num gold 1s:     ", sum(gold == 1 for _, gold in data))

    return correct / n


def train(train_data,
          dev_data,
          model,
          num_classes,
          model_save_dir,
          num_iterations,
          model_file_prefix,
          learning_rate,
          batch_size,
          run_scheduler=False,
          gpu=False,
          clip=None,
          max_len=-1,
          debug=0,
          dropout=0,
          word_dropout=0,
          patience=1000):
    """ Train a model on all the given docs """

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = NLLLoss(None, False)

    enable_gradient_clipping(model, clip)

    if dropout:
        dropout = torch.nn.Dropout(dropout)
    else:
        dropout = None

    debug_print = int(100 / batch_size) + 1

    writer = None

    if model_save_dir is not None:
        writer = SummaryWriter(os.path.join(model_save_dir, "logs"))

    if run_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    best_dev_loss = 100000000
    best_dev_loss_index = -1
    best_dev_acc = -1
    start_time = monotonic()

    for it in range(num_iterations):
        np.random.shuffle(train_data)

        loss = 0.0
        i = 0
        for batch in shuffled_chunked_sorted(train_data, batch_size):
            batch_obj = Batch([x[0] for x in batch], model.embeddings, to_cuda(gpu), word_dropout, max_len)
            gold = [x[1] for x in batch]
            loss += torch.sum(
                train_batch(model, batch_obj, num_classes, gold, optimizer, loss_function, gpu, debug, dropout)
            )

            if i % debug_print == (debug_print - 1):
                print(".", end="", flush=True)
            i += 1

        if writer is not None:
            for name, param in model.named_parameters():
                writer.add_scalar("parameter_mean/" + name,
                                  param.data.mean(),
                                  it)
                writer.add_scalar("parameter_std/" + name, param.data.std(), it)
                if param.grad is not None:
                    writer.add_scalar("gradient_mean/" + name,
                                      param.grad.data.mean(),
                                      it)
                    writer.add_scalar("gradient_std/" + name,
                                      param.grad.data.std(),
                                      it)

            writer.add_scalar("loss/loss_train", loss, it)

        dev_loss = 0.0
        i = 0
        for batch in chunked_sorted(dev_data, batch_size):
            batch_obj = Batch([x[0] for x in batch], model.embeddings, to_cuda(gpu))
            gold = [x[1] for x in batch]
            dev_loss += torch.sum(compute_loss(model, batch_obj, num_classes, gold, loss_function, gpu, debug).data)

            if i % debug_print == (debug_print - 1):
                print(".", end="", flush=True)

            i += 1

        if writer is not None:
            writer.add_scalar("loss/loss_dev", dev_loss, it)
        print("\n")

        finish_iter_time = monotonic()
        train_acc = evaluate_accuracy(model, train_data[:1000], batch_size, gpu)
        dev_acc = evaluate_accuracy(model, dev_data, batch_size, gpu)

        print(
            "iteration: {:>7,} train time: {:>9,.3f}m, eval time: {:>9,.3f}m "
            "train loss: {:>12,.3f} train_acc: {:>8,.3f}% "
            "dev loss: {:>12,.3f} dev_acc: {:>8,.3f}%".format(
                it,
                (finish_iter_time - start_time) / 60,
                (monotonic() - finish_iter_time) / 60,
                loss / len(train_data),
                train_acc * 100,
                dev_loss / len(dev_data),
                dev_acc * 100
            )
        )

        if dev_loss < best_dev_loss:
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                print("New best acc!")
            print("New best dev!")
            best_dev_loss = dev_loss
            best_dev_loss_index = 0
            if model_save_dir is not None:
                model_save_file = os.path.join(model_save_dir, "{}_{}.pth".format(model_file_prefix, it))
                print("saving model to", model_save_file)
                torch.save(model.state_dict(), model_save_file)
        else:
            best_dev_loss_index += 1
            if best_dev_loss_index == patience:
                print("Reached", patience, "iterations without improving dev loss. Breaking")
                break

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print("New best acc!")
            if model_save_dir is not None:
                model_save_file = os.path.join(model_save_dir, "{}_{}.pth".format(model_file_prefix, it))
                print("saving model to", model_save_file)
                torch.save(model.state_dict(), model_save_file)

        if run_scheduler:
            scheduler.step(dev_loss)

    return model


def main(args):
    print(args)

    pattern_specs = OrderedDict(sorted(([int(y) for y in x.split("-")] for x in args.patterns.split("_")),
                                key=lambda t: t[0]))

    pre_computed_patterns = None

    if args.pre_computed_patterns is not None:
        pre_computed_patterns = read_patterns(args.pre_computed_patterns, pattern_specs)
        pattern_specs = OrderedDict(sorted(pattern_specs.items(), key=lambda t: t[0]))

    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab size:", len(dev_vocab))
    train_vocab = vocab_from_text(args.td)
    print("Train vocab size:", len(train_vocab))
    dev_vocab |= train_vocab

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    num_padding_tokens = max(list(pattern_specs.keys())) - 1

    dev_input, _ = read_docs(args.vd, vocab, num_padding_tokens=num_padding_tokens)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))

    np.random.shuffle(dev_data)
    num_iterations = args.num_iterations

    train_input, _ = read_docs(args.td, vocab, num_padding_tokens=num_padding_tokens)
    train_labels = read_labels(args.tl)

    print("training instances:", len(train_input))

    num_classes = len(set(train_labels))

    # truncate data (to debug faster)
    train_data = list(zip(train_input, train_labels))
    np.random.shuffle(train_data)

    print("num_classes:", num_classes)

    if n is not None:
        train_data = train_data[:n]
        dev_data = dev_data[:n]

    if args.use_rnn:
        rnn = Rnn(word_dim,
                  args.hidden_dim,
                  cell_type=LSTM,
                  gpu=args.gpu)
    else:
        rnn = None

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
                                  args.bias_scale_param,
                                  args.gpu,
                                  rnn,
                                  pre_computed_patterns,
                                  args.no_sl,
                                  args.shared_sl,
                                  args.no_eps,
                                  args.eps_scale,
                                  args.self_loop_scale)

    if args.gpu:
        model.to_cuda(model)

    model_file_prefix = 'model'
    # Loading model
    if args.input_model is not None:
        state_dict = torch.load(args.input_model)
        model.load_state_dict(state_dict)
        model_file_prefix = 'model_retrained'

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
          args.batch_size,
          args.scheduler,
          args.gpu,
          args.clip,
          args.max_doc_len,
          args.debug,
          args.dropout,
          args.word_dropout,
          args.patience)

    return 0


def read_patterns(ifile, pattern_specs):
    with open(ifile, encoding='utf-8') as ifh:
        pre_computed_patterns = [l.rstrip().split() for l in ifh if len(l.rstrip())]

    for p in pre_computed_patterns:
        l = len(p) + 1

        if l not in pattern_specs:
            pattern_specs[l] = 1
        else:
            pattern_specs[l] += 1

    return pre_computed_patterns


def soft_pattern_arg_parser():
    """ CLI args related to SoftPatternsClassifier """
    p = ArgumentParser(add_help=False,
                       parents=[lstm_arg_parser(), mlp_arg_parser()])
    p.add_argument("-u", "--use_rnn", help="Use an RNN underneath soft-patterns", action="store_true")
    p.add_argument("-p", "--patterns",
                   help="Pattern lengths and numbers: an underscore separated list of length-number pairs",
                   default="5-50_4-50_3-50_2-50")
    p.add_argument("--maxplus",
                   help="Use max-plus semiring instead of plus-times",
                   default=False, action='store_true')
    p.add_argument("--maxtimes",
                   help="Use max-times semiring instead of plus-times",
                   default=False, action='store_true')
    p.add_argument("--bias_scale_param",
                   help="Scale bias term by this parameter",
                   default=0.1, type=float)
    p.add_argument("--eps_scale",
                   help="Scale epsilon by this parameter",
                   default=None, type=float)
    p.add_argument("--self_loop_scale",
                   help="Scale self_loop by this parameter",
                   default=None, type=float)
    p.add_argument("--no_eps", help="Don't use epsilon transitions", action='store_true')
    p.add_argument("--no_sl", help="Don't use self loops", action='store_true')
    p.add_argument("--shared_sl",
                   help="Share main path and self loop parameters, where self loops are discounted by a self_loop_parameter. "+
                           str(SHARED_SL_PARAM_PER_STATE_PER_PATTERN)+
                           ": one parameter per state per pattern, "+str(SHARED_SL_SINGLE_PARAM)+
                           ": a global parameter.", type=int, default=0)

    return p




def training_arg_parser():
    """ CLI args related to training models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("-i", "--num_iterations", help="Number of iterations", type=int, default=10)
    p.add_argument("--patience", help="Patience parameter (for early stopping)", type=int, default=30)
    p.add_argument("-m", "--model_save_dir", help="where to save the trained model")
    p.add_argument("-r", "--scheduler", help="Use reduce learning rate on plateau schedule", action='store_true')
    p.add_argument("-w", "--word_dropout", help="Use word dropout", type=float, default=0)
    p.add_argument("--td", help="Train data file", required=True)
    p.add_argument("--tl", help="Train labels file", required=True)
    p.add_argument("--pre_computed_patterns", help="File containing pre-computed patterns")
    p.add_argument("-l", "--learning_rate", help="Adam Learning rate", type=float, default=1e-3)
    p.add_argument("--clip", help="Gradient clipping", type=float, default=None)
    p.add_argument("--debug", help="Debug", type=int, default=0)
    return p

def general_arg_parser():
    """ CLI args related to training and testing models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("-b", "--batch_size", help="Batch size", type=int, default=1)
    p.add_argument("--max_doc_len",
                   help="Maximum doc length. For longer documents, spans of length max_doc_len will be randomly "
                        "selected each iteration (-1 means no restriction)",
                   type=int, default=-1)
    p.add_argument("-s", "--seed", help="Random seed", type=int, default=100)
    p.add_argument("-n", "--num_train_instances", help="Number of training instances", type=int, default=None)
    p.add_argument("--vd", help="Validation data file", required=True)
    p.add_argument("--vl", help="Validation labels file", required=True)
    p.add_argument("--input_model", help="Input model (to run test and not train)")
    p.add_argument("-t", "--dropout", help="Use dropout", type=float, default=0)
    p.add_argument("-g", "--gpu", help="Use GPU", action='store_true')
    p.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)

    return p


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[soft_pattern_arg_parser(), training_arg_parser(), general_arg_parser()])
    sys.exit(main(parser.parse_args()))
