#!/usr/bin/env python
"""
Text classification baseline models.
"""
import argparse
from soft_patterns import fixed_var, train
import numpy as np
import os
import torch
from torch import FloatTensor, cuda
from torch.functional import stack
from torch.nn import Module, Dropout2d

from data import read_embeddings, read_docs, read_labels, vocab_from_text
from mlp import MLP


# TODO: word dropout
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
                 vocab,
                 gpu=False,
                 dropout=0.1):
        super(DanClassifier, self).__init__()
        self.vocab = vocab
        self.embeddings = fixed_var(FloatTensor(embeddings), gpu)
        self.word_dim = len(embeddings[0])
        self.dtype = FloatTensor
        if gpu:
            self.embeddings.cuda()
            self.dtype = cuda.FloatTensor
        self.gpu = gpu
        self.mlp = MLP(self.word_dim,
                       mlp_hidden_dim,
                       num_mlp_layers,
                       num_classes,
                       dropout,
                       legacy=False)
        self.dropout = Dropout2d(dropout) if dropout else None
        print("# params:", sum(p.nelement() for p in self.parameters()))

    def forward(self, doc):
        """ Average all word vectors in the doc, and feed into an MLP """
        n = len(doc)
        doc_vectors = stack([self.embeddings[i] for i in doc])
        if self.dropout:
            # dropout entire words at a time
            doc_vectors = self.dropout(doc_vectors.view(n, 1, self.word_dim)).view(n, self.word_dim)
        avg_word_vector = torch.sum(doc_vectors, dim=0) / n
        return self.mlp.forward(avg_word_vector)

    def predict(self, doc):
        old_training = self.training
        self.train(False)
        output = self.forward(doc).data
        _, am = torch.max(output, 0)
        self.train(old_training)
        return int(am[0])


# TODO: refactor duplicate code with soft_patterns.py
def main(args):
    print(args)
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

    if args.td is None or args.tl is None:
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

    print("num_classes:", num_classes)

    if n is not None:
        train_data = train_data[:n]
        dev_data = dev_data[:n]

    dropout = None if args.td is None else args.dropout

    model = DanClassifier(mlp_hidden_dim,
                          num_mlp_layers,
                          num_classes,
                          embeddings,
                          reverse_vocab,
                          args.gpu,
                          dropout)

    if args.gpu:
        model.cuda()

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
          args.scheduler,
          args.gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)
    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=100)
    parser.add_argument("-i", "--num_iterations", help="Number of iterations", type=int, default=10)
    parser.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", type=int, default=10)
    parser.add_argument("-y", "--num_mlp_layers", help="Number of MLP layers", type=int, default=2)
    parser.add_argument("-n", "--num_train_instances", help="Number of training instances", type=int, default=None)
    parser.add_argument("-m", "--model_save_dir", help="where to save the trained model")
    parser.add_argument("-r", "--scheduler", help="Use reduce learning rate on plateau schedule", action='store_true')
    parser.add_argument("-g", "--gpu", help="Use GPU", action='store_true')
    parser.add_argument("-t", "--dropout", help="Use dropout", type=float, default=0)
    parser.add_argument("--input_model", help="Input model (to run test and not train)")
    parser.add_argument("--td", help="Train data file")
    parser.add_argument("--tl", help="Train labels file")
    parser.add_argument("--vd", help="Validation data file", required=True)
    parser.add_argument("--vl", help="Validation labels file", required=True)
    parser.add_argument("-l", "--learning_rate", help="Adam Learning rate", type=float, default=1e-3)

    main(parser.parse_args())
