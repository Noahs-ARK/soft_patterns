""" Utility functions """
import numpy as np


def identity(x):
    return x


def nub(xs):
    """ Removes duplicates, maintaining original order. """
    return nub_by(xs, identity)


def nub_by(xs, key):
    """ Removes elements with duplicate keys, maintaining original order. """
    seen = set()

    def check_and_add(x):
        k = key(x)
        if k not in seen:
            seen.add(k)
            return True
        return False

    return (x for x in xs if check_and_add(x))


def chunked(xs, chunk_size):
    """ Splits a list into `chunk_size`-sized pieces. """
    xs = list(xs)
    return [
        xs[i:i + chunk_size]
        for i in range(0, len(xs), chunk_size)
    ]


def decreasing_length(xs):
    return sorted(list(xs), key=lambda x: len(x[0]), reverse=True)


def chunked_sorted(xs, chunk_size):
    return chunked(decreasing_length(xs), chunk_size)


def shuffled_chunked_sorted(xs, chunk_size):
    """ Splits a list into `chunk_size`-sized pieces. """
    chunks = chunked_sorted(xs, chunk_size)
    np.random.shuffle(chunks)
    return chunks


def right_pad(xs, min_len, pad_element):
    """
    Appends `pad_element`s to `xs` so that it has length `min_len`.
    No-op if `len(xs) >= min_len`.
    """
    return xs + [pad_element] * (min_len - len(xs))


def to_cuda(gpu):
    return (lambda v: v.cuda()) if gpu else identity
