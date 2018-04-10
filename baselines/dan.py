#!/usr/bin/env python3 -u
"""
Text classification baseline model "DAN".

example usage:
./baselines/dan.py \
        -e ${wordvec_file} \
        --td ${sst_dir}/train.data \
        --tl ${sst_dir}/train.labels \
        --vd ${sst_dir}/dev.data \
        --vl ${sst_dir}/dev.labels \
        -l 0.001 \
        -i 70 \
        -m ./experiments/dan \
        -n 100
"""
import argparse
import sys; sys.path.append(".")
from soft_patterns import train, training_arg_parser, general_arg_parser
from util import to_cuda
import numpy as np
import os
import torch
from torch.nn import Module

from data import read_embeddings, read_docs, read_labels, vocab_from_text
from mlp import MLP, mlp_arg_parser


class DanClassifier(Module):
    """
    A text classification model based on
    "Deep Unordered Composition Rivals Syntactic Methods for Text Classification"
    Iyyer et al., ACL 2015
    """
    def __init__(self,
                 mlp_hidden_dim,
                 num_mlp_layers,
                 num_classes,
                 embeddings,
                 gpu=False):
        super(DanClassifier, self).__init__()
        self.to_cuda = to_cuda(gpu)
        self.embeddings = embeddings
        self.word_dim = len(embeddings[0])
        self.mlp = MLP(self.word_dim,
                       mlp_hidden_dim,
                       num_mlp_layers,
                       num_classes)
        print("# params:", sum(p.nelement() for p in self.parameters()))

    def forward(self, batch, debug=0, dropout=None):
        """ Average all word vectors in the doc, and feed into an MLP """
        docs_vectors = [
            torch.index_select(batch.embeddings_matrix, 1, doc)
            for doc in batch.docs
        ]

        # don't need to mask docs because padding vector is 0, won't change sum
        word_vector_sum = torch.sum(torch.stack(docs_vectors), dim=2)
        word_vector_avg = \
            torch.div(word_vector_sum.t(),
                      torch.autograd.Variable(batch.doc_lens.float())).t()

        return self.mlp.forward(word_vector_avg)

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

    model = DanClassifier(args.mlp_hidden_dim,
                          args.num_mlp_layers,
                          num_classes,
                          embeddings,
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
          dropout=args.dropout,
          word_dropout=args.word_dropout,
          patience=args.patience)



if __name__ == '__main__':
    parser = \
        argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                parents=[training_arg_parser(), general_arg_parser()])
    main(parser.parse_args())
