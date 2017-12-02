#!/usr/bin/env python3 -u
"""
Text classification baseline model "DAN".

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
from soft_patterns import train, to_cuda, training_arg_parser
import numpy as np
import os
import torch
from torch.autograd import Variable
from torch.nn import Module, LSTM, Parameter
from data import read_embeddings, read_docs, read_labels, vocab_from_text
from mlp import MLP


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
                 gpu=False,
                 dropout=0.1):
        super(AveragingRnnClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.to_cuda = to_cuda(gpu)
        self.embeddings = embeddings
        self.word_dim = len(embeddings[0])
        self.cell_type = cell_type
        self.rnn = self.cell_type(input_size=self.word_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=1,
                                  dropout=dropout,
                                  bidirectional=True)
        self.num_directions = 2  # We're a *bi*LSTM
        self.start_hidden_state = \
            Parameter(self.to_cuda(
                torch.randn(self.num_directions, 1, self.hidden_dim)
            ))
        self.start_cell_state = \
            Parameter(self.to_cuda(
                torch.randn(self.num_directions, 1, self.hidden_dim)
            ))
        self.mlp = MLP(self.num_directions * self.hidden_dim,
                       mlp_hidden_dim,
                       num_mlp_layers,
                       num_classes)
        print("# params:", sum(p.nelement() for p in self.parameters()))

    def forward(self, batch, debug=0, dropout=None):
        """
        Run a biLSTM over the batch of docs, average the hidden states, and
        feed into an MLP.
        """
        b = len(batch.docs)
        docs_vectors = [
            torch.index_select(batch.embeddings_matrix, 1, doc).t()
            for doc in batch.docs
        ]

        # run the biLSTM
        starts = (
            self.start_hidden_state.expand(self.num_directions, b, self.hidden_dim),
            self.start_cell_state.expand(self.num_directions, b, self.hidden_dim)
        )
        outs, _ = self.rnn(torch.stack(docs_vectors, dim=1), starts)

        # average all the hidden states
        # TODO: mask so hidden states past end of doc aren't considered
        outs_sum = torch.sum(outs, dim=0)  # avg each doc
        outs_avg = torch.div(
            outs_sum,
            Variable(batch.doc_lens.float().view(b, 1)).expand(b, self.num_directions * self.hidden_dim)
        )
        return self.mlp.forward(outs_avg)

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

    dev_input, dev_text = read_docs(args.vd, vocab, 1)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))

    np.random.shuffle(dev_data)
    train_input, _ = read_docs(args.td, vocab, 1)
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
                                   gpu=args.gpu,
                                   dropout=dropout)

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
          dropout=args.dropout,
          word_dropout=args.word_dropout,
          patience=args.patience)


def lstm_arg_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)
    p.add_argument("-g", "--gpu", help="Use GPU", action='store_true')
    p.add_argument("-t", "--dropout", help="Use dropout", type=float, default=0)
    p.add_argument("--hidden_dim", help="RNN hidden dimension", type=int, default=100)
    p.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", type=int, default=10)
    p.add_argument("-y", "--num_mlp_layers", help="Number of MLP layers", type=int, default=2)
    # p.add_argument("--gru", help="Use GRU cells instead of LSTM cells", action='store_true')
    return p


if __name__ == '__main__':
    parser = \
        argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                parents=[lstm_arg_parser(), training_arg_parser()])
    main(parser.parse_args())
