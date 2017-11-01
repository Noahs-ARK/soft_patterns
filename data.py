""" Reading in data files """
import string

import numpy as np


UNK_TOKEN = "*UNK*"


class Vocab:
    def __init__(self, names, default=UNK_TOKEN):
        self.default = default
        self.names = [self.default] + list(set(names) - set([self.default]))
        self.index = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, index):
        """ Lookup name given index. """
        return self.names[index] if 0 < index < len(self.names) else self.default

    def __call__(self, name):
        """ Lookup index given name. """
        return self.index.get(name, 0)

    def numberize(self, doc):
        """ Replace each name in doc with its index. """
        return [self(token) for token in doc]

    def denumberize(self, doc):
        """ Replace each index in doc with its name. """
        return [self[idx] for idx in doc]

    @staticmethod
    def from_docs(docs, default=UNK_TOKEN):
        return Vocab((i for doc in docs for i in doc), default=default)


def read_embeddings(filename, train_vocab=None):
    vocab = {UNK_TOKEN:  0}

    vecs = []
    printable = set(string.printable)

    print("Reading", filename)
    with open(filename, encoding='utf-8') as input_file:
        first_line = input_file.readline()

        e = first_line.rstrip().split()

        # Header line
        if len(e) == 2:
            dim = int(e[1])
            vecs.append(np.zeros(dim))  # UNK embedding (at idx 0) is all zeros
        else:
            dim = len(e) - 1
            vecs.append(np.zeros(dim))
            e = first_line.rstrip().split(" ", 1)
            add_vec(e, vecs, vocab, train_vocab)

        for line in input_file:
            e = line.rstrip().split(" ", 1)
            word = e[0]

            # TODO: this is for debugging only
            # if len(vocab) == 5000:
            #     break

            good = 1
            for i in list(word):
                if i not in printable:
                    good = 0
                    break

            if not good:
                # print(n,"is not good")
                continue

            add_vec(e, vecs, vocab, train_vocab)

    print("Done reading", len(vecs), "vectors of dimension", dim)
    reverse_vocab = {
        i: name for name, i in vocab.items()
    }
    return vocab, reverse_vocab, vecs, dim


def add_vec(v, vecs, vocab, train_vocab=None):
    w = v[0]
    if train_vocab is not None and w not in train_vocab:
        return
    vocab[w] = len(vocab)
    vec = np.fromstring(v[1], dtype=float, sep=' ')
    vecs.append(vec / np.linalg.norm(vec))


def read_docs(filename, vocab):
    with open(filename, encoding='utf-8') as input_file:
        train_words = [line.rstrip().split() for line in input_file]
        return [[vocab[w] if w in vocab else 0 for w in doc] for doc in train_words], train_words


def read_labels(filename):
    with open(filename) as input_file:
        return [int(line.rstrip()) for line in input_file]


def vocab_from_text(filename):
    vocab = set()
    with open(filename, encoding='utf-8') as input_file:
        for line in input_file:
            train_words = set(line.rstrip().split())

            vocab |= train_words

    print("Vocab from text", len(vocab))
    return vocab