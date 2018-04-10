import argparse

import torch
from torch.nn import Parameter, Module, LSTM
from torch.nn.utils.rnn import pack_padded_sequence

from util import to_cuda


class Rnn(Module):
    """ A BiLSTM or BiGRU """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 cell_type=LSTM,
                 gpu=False):
        super(Rnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.to_cuda = to_cuda(gpu)
        self.input_dim = input_dim
        self.cell_type = cell_type
        self.rnn = self.cell_type(input_size=self.input_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=1,
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

    def forward(self, batch, debug=0, dropout=None):
        """
        Run a biLSTM over the batch of docs, return their hidden states (padded).
        """
        b = len(batch.docs)
        docs_vectors = [
            torch.index_select(batch.embeddings_matrix, 1, doc).t()
            for doc in batch.docs
        ]
        # Assumes/requires that `batch.docs` is sorted by decreasing doc length.
        # This gets done in `chunked_sorted`.
        packed = pack_padded_sequence(
            torch.stack(docs_vectors, dim=1),
            lengths=list(batch.doc_lens)
        )

        # run the biLSTM
        starts = (
            self.start_hidden_state.expand(self.num_directions, b, self.hidden_dim).contiguous(),
            self.start_cell_state.expand(self.num_directions, b, self.hidden_dim).contiguous()
        )
        outs, _ = self.rnn(packed, starts)

        return outs


def lstm_arg_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--hidden_dim", help="RNN hidden dimension", type=int, default=100)
    # p.add_argument("--gru", help="Use GRU cells instead of LSTM cells", action='store_true')
    return p
