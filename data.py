""" Reading in data files """
import string

import numpy as np


def read_embeddings(file):
    vocab = dict()
    vocab["*UNK*"] = 0

    vecs = []
    printable = set(string.printable)

    print("Reading", file)
    with open(file, encoding='utf-8') as ifh:
        first_line = ifh.readline()

        e = first_line.rstrip().split()

        # Header line
        if len(e) == 2:
            dim = int(e[1])
            vecs.append(np.zeros(dim))
        else:
            dim = len(e) - 1
            vecs.append(np.zeros(dim))

            vocab[e[0]] = 1
            vecs.append(np.fromstring(" ".join(e[1:]), dtype=float, sep=' '))

        for l in ifh:
            e = l.rstrip().split(" ", 1)
            word = e[0]

            if len(vocab) == 10:
                break
            good = 1
            for i in list(word):
                if i not in printable:
                    good = 0
                    break

            if not good:
                # print(n,"is not good")
                continue

            vocab[e[0]] = len(vocab)
            vecs.append(np.fromstring(e[1], dtype=float, sep=' '))

    print("Done reading", len(vecs), "vectors of dimension", dim)
    return vocab, vecs, dim


def read_sentences(filename, vocab):
    with open(filename, encoding='utf-8') as ifh:
        train_words = [x.rstrip().split() for x in ifh]
        train_data = [[vocab[w] if w in vocab else 0 for w in sent] for sent in train_words]
    return train_data


def read_labels(filename):
    with open(filename) as ifh:
        train_labels = [int(x.rstrip()) for x in ifh]
    return train_labels
