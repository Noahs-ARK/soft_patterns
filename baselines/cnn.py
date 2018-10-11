#!/usr/bin/env python -u
"""
Text classification baseline: Convolutional Neural Network.

example usage:
./baselines/cnn.py \
        -e ${wordvec_file} \
        --td ${sst_dir}/train.data \
        --tl ${sst_dir}/train.labels \
        --vd ${sst_dir}/dev.data \
        --vl ${sst_dir}/dev.labels \
        -l 0.001 \
        -i 70 \
        -m ./experiments/cnn \
        -n 100 \
        --num_cnn_layers 2 \
        --cnn_hidden_dim 100 \
        --num_mlp_layers 1 \
        --mlp_hidden_dim 100 \
        --window_size 4 \
        --batch_size 3
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys; sys.path.append(".")
from soft_patterns import train, training_arg_parser, general_arg_parser
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import relu
import numpy as np
import os
import torch
from torch.autograd import Variable
from torch.nn import Module, Conv1d
from data import read_embeddings, read_docs, read_labels, vocab_from_text
from mlp import MLP, mlp_arg_parser
from util import to_cuda

NEG_INF = float("-inf")


def max_pool_seq(packed_seq):
    """ Given a PackedSequence, max-pools each sequence in the batch. """
    # unpack
    padded, _ = pad_packed_sequence(packed_seq, padding_value=NEG_INF)  # size: (max_doc_len, batch_size, *hidden_dims)
    # max-pool each doc
    maxes, _ = torch.max(padded, dim=0)  # size: (batch_size, *hidden_dims)
    return maxes


def sum_pool_seq(packed_seq):
    """ Given a PackedSequence, sum-pools each sequence in the batch. """
    # unpack
    padded, lens = pad_packed_sequence(packed_seq)  # size: (max_doc_len, batch_size, *hidden_dims)
    # sum each doc
    return torch.sum(padded, dim=0)  # size: (batch_size, *hidden_dims)


def average_pool_seq(packed_seq):
    """ Given a PackedSequence, average-pools each sequence in the batch. """
    # unpack
    padded, lens = pad_packed_sequence(packed_seq)  # size: (max_doc_len, batch_size, *hidden_dims)
    b = len(lens)
    # sum and normalize by doc length
    sums = torch.sum(padded, dim=0)  # size: (batch_size, *hidden_dims)
    # size: (batch_size, *hidden_dims)
    return torch.div(
            sums,
            Variable(torch.FloatTensor(lens).view(b, 1)).expand(*sums.size())
        )


# copy-pasted from torch/nn/utils/rnn.py, added parameter for default padding
def pad_packed_sequence(sequence, batch_first=False, padding_value=None):
    """Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Variable's data will be of size TxBx*, where T is the length
    of the longest sequence and B is the batch size. If ``batch_first`` is True,
    the data will be transposed into BxTx* format.

    Batch elements will be ordered decreasingly by their length.

    Arguments:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if True, the output will be in BxTx*
            format.
        padding_value (float, optional): the value with which to pad shorter
            sequences in the batch (default is 0).

    Returns:
        Tuple of Variable containing the padded sequence, and a list of lengths
        of each sequence in the batch.
    """
    var_data, batch_sizes = sequence
    max_batch_size = batch_sizes[0]
    output = var_data.data.new(len(batch_sizes), max_batch_size, *var_data.size()[1:])
    if padding_value is not None:
        output.fill_(padding_value)
    else:
        output.zero_()
    output = Variable(output)

    lengths = []
    data_offset = 0
    prev_batch_size = batch_sizes[0]
    for i, batch_size in enumerate(batch_sizes):
        output[i, :batch_size] = var_data[data_offset:data_offset + batch_size]
        data_offset += batch_size

        dec = prev_batch_size - batch_size
        if dec > 0:
            lengths.extend((i,) * dec)
        prev_batch_size = batch_size
    lengths.extend((i + 1,) * batch_size)
    lengths.reverse()

    if batch_first:
        output = output.transpose(0, 1)
    return output, lengths


class Cnn(Module):
    """
    A model that runs a deep CNN over a document.
    If `num_layers > 1`, each window is fed into an MLP with `num_layers - 1`
    layers, each layer having dimension `output_dim`.
    Returns a PackedPaddedSequence that when unpacked has size:
    ((max_doc_len - window_size + 1), batch_size, output_dim).
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_layers,
                 output_dim,
                 window_size,
                 gpu=False):
        super(Cnn, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.window_size = window_size
        self.gpu = gpu
        if not num_layers <= 1:
            self.cnn = \
                Conv1d(input_dim,
                       hidden_dim,
                       window_size)
            self.mlp = \
                MLP(hidden_dim,
                    hidden_dim,
                    num_layers - 1,
                    output_dim)
        else:
            self.cnn = \
                Conv1d(input_dim,
                       output_dim,
                       window_size)
            self.mlp = None

    def forward(self, batch, debug=0, dropout=None):
        docs = batch.docs
        doc_lens = batch.doc_lens
        b = len(docs)
        max_doc_len = max(list(doc_lens))
        docs_vectors = \
            torch.stack(
                [
                    torch.index_select(batch.embeddings_matrix, 1, doc)
                    for doc in docs
                ],
                dim=0
            )
        # right pad, so docs are at least as long as self.window_size
        doc_lens = [max(self.window_size, l) for l in doc_lens]
        if max_doc_len < self.window_size:
            print("max doc length {} is smaller than window size {}".format(max_doc_len, self.window_size))
            docs_vectors = \
                torch.cat(
                    (docs_vectors, torch.zeros(b, self.window_size - max_doc_len)),
                    dim=1
                )
            max_doc_len = self.window_size
        cnn_outs = self.cnn.forward(docs_vectors)  # size: (b, hidden_dim, max_doc_len - window_size + 1)
        num_windows_per_doc = cnn_outs.size()[2]
        assert(num_windows_per_doc == max_doc_len - self.window_size + 1)

        if dropout is not None:
            cnn_outs = dropout(cnn_outs)

        if self.num_layers <= 1:
            result = cnn_outs.permute(2, 0, 1)
        else:
            # reshape so all windows can be passed into MLP
            cnn_outs = \
                cnn_outs.transpose(1, 2).contiguous().view(b * num_windows_per_doc, self.hidden_dim)
            # run MLP on all windows
            cnn_outs = relu(cnn_outs)
            mlp_outs = self.mlp.forward(cnn_outs)
            # size: (max_doc_len - window_size + 1, b, hidden_dim)
            result = mlp_outs.view(b, num_windows_per_doc, self.hidden_dim).transpose(0, 1)
        lengths = [max(0, l - self.window_size + 1) for l in doc_lens]
        # pack to get rid of the parts that are past the end of the doc
        return pack_padded_sequence(
            result,
            lengths=lengths
        )


class PooledCnnClassifier(Module):
    """
    A text classification model that runs a CNN over a document,
    pools the hidden states, then feeds that into an MLP.
    `hidden_dim` is used for the hidden layers and output layer of the CNN,
    as well as the hidden layers of the MLP.
    """
    def __init__(self,
                 window_size,
                 num_cnn_layers,
                 cnn_hidden_dim,
                 num_mlp_layers,
                 mlp_hidden_dim,
                 num_classes,
                 embeddings,
                 pooling=max_pool_seq,
                 gpu=False):
        super(PooledCnnClassifier, self).__init__()
        self.window_size = window_size
        self.hidden_dim = cnn_hidden_dim
        self.num_cnn_layers = num_cnn_layers
        self.num_mlp_layers = num_mlp_layers
        self.num_classes = num_classes
        self.embeddings = embeddings
        self.pooling = pooling
        self.cnn = \
            Cnn(len(embeddings[0]),
                cnn_hidden_dim,
                num_cnn_layers,
                cnn_hidden_dim,
                window_size,
                gpu=gpu)
        self.mlp = \
            MLP(cnn_hidden_dim,
                mlp_hidden_dim,
                num_mlp_layers,
                num_classes)

        self.to_cuda = to_cuda(gpu)
        print("# params:", sum(p.nelement() for p in self.parameters()))

    def forward(self, batch, debug=0, dropout=None):
        """
        Run a CNN over the batch of docs, average the hidden states, and
        feed into an MLP.
        """
        # run CNN
        cnn_outs = self.cnn.forward(batch, debug=debug, dropout=dropout)
        # pool the hidden states
        pooled = self.pooling(cnn_outs)  # size (b, hidden_dim)
        # run MLP
        return self.mlp.forward(pooled)  # size (b, output_dim)

    def predict(self, batch, debug=0):
        old_training = self.training
        self.train(False)
        output = self.forward(batch, debug=debug, dropout=None).data
        _, am = torch.max(output, 1)
        self.train(old_training)
        return [int(x) for x in am]


# TODO: refactor duplicate code with soft_patterns.py
def main(args):
    print(args)
    n = args.num_train_instances
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab:", len(dev_vocab))
    train_vocab = vocab_from_text(args.td)
    print("Train vocab:", len(train_vocab))
    dev_vocab |= train_vocab

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    num_padding_tokens = args.window_size - 1

    dev_input, _ = read_docs(args.vd, vocab, num_padding_tokens=num_padding_tokens)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))

    np.random.shuffle(dev_data)
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

    dropout = None if args.td is None else args.dropout

    pooling = sum_pool_seq if args.pooling == "sum" else (
        average_pool_seq if args.pooling == "avg" else
        max_pool_seq
    )

    model = \
        PooledCnnClassifier(
            args.window_size,
            args.num_cnn_layers,
            args.cnn_hidden_dim,
            args.num_mlp_layers,
            args.mlp_hidden_dim,
            num_classes,
            embeddings,
            pooling=pooling,
            gpu=args.gpu
        )

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
          args.num_iterations,
          model_file_prefix,
          args.learning_rate,
          args.batch_size,
          args.scheduler,
          gpu=args.gpu,
          clip=args.clip,
          max_len=args.max_doc_len,
          debug=args.debug,
          dropout=dropout,
          word_dropout=args.word_dropout,
          patience=args.patience)


def cnn_arg_parser():
    """ CLI args related to the MLP module """
    p = ArgumentParser(add_help=False)
    # we're running out of letters!
    p.add_argument("-c", "--cnn_hidden_dim", help="CNN hidden dimension", type=int, default=200)
    p.add_argument("-x", "--num_cnn_layers", help="Number of MLP layers", type=int, default=2)
    p.add_argument("-z", "--window_size", help="Size of window of CNN", type=int, default=3)
    p.add_argument("-o", "--pooling", help="Type of pooling to use [max, sum, avg]", type=str, default="max")
    return p


def pooling_cnn_arg_parser():
    p = ArgumentParser(add_help=False,
                       parents=[cnn_arg_parser(), mlp_arg_parser()])
    return p


if __name__ == '__main__':
    parser = \
        ArgumentParser(description=__doc__,
                       formatter_class=ArgumentDefaultsHelpFormatter,
                       parents=[pooling_cnn_arg_parser(), training_arg_parser(), general_arg_parser()])
    main(parser.parse_args())
