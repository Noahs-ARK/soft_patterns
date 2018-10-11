#!/usr/bin/env python -u
"""
Text classification baseline: bi-LSTM.

example usage:
./baselines/lstm.py \
        -e ${wordvec_file} \
        --td ${sst_dir}/train.data \
        --tl ${sst_dir}/train.labels \
        --vd ${sst_dir}/dev.data \
        --vl ${sst_dir}/dev.labels \
        -l 0.001 \
        -i 70 \
        -m ./experiments/lsmt \
        -n 100 \
        --hidden_dim 100
        # --gru
"""

import argparse
import sys; sys.path.append(".")
from soft_patterns import train, training_arg_parser, general_arg_parser
from torch.nn.utils.rnn import pad_packed_sequence
from rnn import lstm_arg_parser, Rnn
import numpy as np
import os
import torch
from torch.autograd import Variable
from torch.nn import Module, LSTM
from data import read_embeddings, read_docs, read_labels, vocab_from_text
from mlp import MLP, mlp_arg_parser
from util import to_cuda


class AveragingRnnClassifier(Module):
    """
    A text classification model that runs a biLSTM (or biGRU) over a document,
    averages the hidden states, then feeds that into an MLP.
    """
    def __init__(self,
                 hidden_dim,
                 mlp_hidden_dim,
                 num_mlp_layers,
                 num_classes,
                 embeddings,
                 cell_type=LSTM,
                 gpu=False):
        super(AveragingRnnClassifier, self).__init__()
        self.embeddings = embeddings
        self.rnn = \
            Rnn(len(embeddings[0]),
                hidden_dim,
                cell_type=cell_type,
                gpu=gpu)
        self.mlp = \
            MLP(self.rnn.num_directions * self.rnn.hidden_dim,
                mlp_hidden_dim,
                num_mlp_layers,
                num_classes)

        self.to_cuda = to_cuda(gpu)
        print("# params:", sum(p.nelement() for p in self.parameters()))

    def forward(self, batch, debug=0, dropout=None):
        """
        Run a biLSTM over the batch of docs, average the hidden states, and
        feed into an MLP.
        """
        b = len(batch.docs)
        outs = self.rnn.forward(batch,
                                debug=debug,
                                dropout=dropout)
        padded, _ = pad_packed_sequence(outs)  # size: (max_doc_len, b, 2 * hidden_dim)

        if dropout is not None:
            padded = dropout(padded)
        # average all the hidden states
        outs_sum = torch.sum(padded, dim=0)  # size: (b, 2 * hidden_dim)
        outs_avg = torch.div(
            outs_sum,
            Variable(batch.doc_lens.float().view(b, 1)).expand(b, self.rnn.num_directions * self.rnn.hidden_dim)
        )  # size: (b, 2 * hidden_dim)
        return self.mlp.forward(outs_avg)  # size: (b, num_classes)

    def predict(self, batch, debug=0):
        old_training = self.training
        self.train(False)
        output = self.forward(batch, debug=debug).data
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

    num_padding_tokens = 1

    dev_input, dev_text = read_docs(args.vd, vocab, num_padding_tokens=num_padding_tokens)
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

    # TODO: GRU doesn't work yet
    cell_type = LSTM  # GRU if args.gru else LSTM

    model = AveragingRnnClassifier(args.hidden_dim,
                                   args.mlp_hidden_dim,
                                   args.num_mlp_layers,
                                   num_classes,
                                   embeddings,
                                   cell_type=cell_type,
                                   gpu=args.gpu)

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
          debug=args.debug,
          dropout=dropout,
          word_dropout=args.word_dropout,
          patience=args.patience)


if __name__ == '__main__':
    parser = \
        argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                parents=[lstm_arg_parser(), mlp_arg_parser(), training_arg_parser(), general_arg_parser()])
    main(parser.parse_args())
