""" Reading in data files """
import string

import numpy as np


def read_embeddings(filename):
    vocab = {"*UNK*":  0}

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
            add_vec(e, vecs, vocab)

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

            add_vec(e, vecs, vocab)

    print("Done reading", len(vecs), "vectors of dimension", dim)
    reverse_vocab = {
        i: name for name, i in vocab.items()
    }
    return vocab, reverse_vocab, vecs, dim


def add_vec(v, vecs, vocab):
    vocab[v[0]] = len(vocab)
    vec = np.fromstring(v[1], dtype=float, sep=' ')
    vecs.append(vec / np.linalg.norm(vec))


def read_docs(filename, vocab):
    with open(filename, encoding='utf-8') as input_file:
        train_words = [line.rstrip().split() for line in input_file]
        return [[vocab[w] if w in vocab else 0 for w in doc] for doc in train_words], train_words


def read_labels(filename):
    with open(filename) as input_file:
        return [int(line.rstrip()) for line in input_file]
