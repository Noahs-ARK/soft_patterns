""" Reading in data files """
from itertools import chain, islice
import string
import numpy as np
from util import nub

PRINTABLE = set(string.printable)
UNK_TOKEN = "*UNK*"


def is_printable(word):
    return all(c in PRINTABLE for c in word)


class Vocab:
    """
    A bimap from name to index.
    Use `vocab[i]` to lookup name for `i`,
    and `vocab(n)` to lookup index for `n`.
    """
    def __init__(self,
                 names,
                 default=UNK_TOKEN):
        self.default = default
        self.names = list(nub(chain([default], names)))
        self.index = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, index):
        """ Lookup name given index. """
        return self.names[index] if 0 < index < len(self.names) else self.default

    def __call__(self, name):
        """ Lookup index given name. """
        return self.index.get(name, 0)

    def __contains__(self, item):
        return item in self.index

    def __len__(self):
        return len(self.names)

    def __or__(self, other):
        return Vocab(self.names + other.names)

    def numberize(self, doc):
        """ Replace each name in doc with its index. """
        return [self(token) for token in doc]

    def denumberize(self, doc):
        """ Replace each index in doc with its name. """
        return [self[idx] for idx in doc]

    @staticmethod
    def from_docs(docs, default=UNK_TOKEN):
        return Vocab((i for doc in docs for i in doc), default=default)


def read_embeddings(filename,
                    fixed_vocab=None,
                    max_vocab_size=None):
    print("Reading", filename)
    dim, has_header = check_dim_and_header(filename)
    unk_vec = np.zeros(dim)  # TODO: something better?
    with open(filename, encoding='utf-8') as input_file:
        if has_header:
            input_file.readline()  # skip over header
        word_vecs = (
            (word, np.fromstring(vec_str, dtype=float, sep=' '))
            for word, vec_str in (
                line.rstrip().split(" ", 1)
                for line in input_file
            )
            if is_printable(word) and (fixed_vocab is None or word in fixed_vocab)
        )
        if max_vocab_size is not None:
            word_vecs = islice(word_vecs, max_vocab_size - 1)
        word_vecs = list(word_vecs)

    print("Done reading", len(word_vecs), "vectors of dimension", dim)
    vocab = Vocab((word for word, _ in word_vecs))

    vecs = [unk_vec] + [vec / np.linalg.norm(vec) for _, vec in word_vecs]

    return vocab, vecs, dim


def check_dim_and_header(filename):
    with open(filename, encoding='utf-8') as input_file:
        first_line = input_file.readline().rstrip().split()
        if len(first_line) == 2:
            return int(first_line[1]), True
        else:
            return len(first_line) - 1, False


def read_docs(filename, vocab):
    with open(filename, encoding='utf-8') as input_file:
        docs = [line.rstrip().split() for line in input_file]
        return [vocab.numberize(doc) for doc in docs], docs


def read_labels(filename):
    with open(filename) as input_file:
        return [int(line.rstrip()) for line in input_file]


def vocab_from_text(filename):
    with open(filename, encoding='utf-8') as input_file:
        return Vocab.from_docs(line.rstrip().split() for line in input_file)
