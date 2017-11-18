#!/usr/bin/env python

import sys
import argparse
import re
from collections import Counter
import numpy as np
from data import read_labels
from sklearn import linear_model
from scipy.sparse import lil_matrix


INDEX_TOKEN = "###INDEX###"

def main(args):
    with open(args.work_dir+"/train.data", encoding="utf-8") as ifh:
        wordcount = Counter(ifh.read().split())

    sum = np.sum(list(wordcount.values()))

    # print(sum)

    wordcount = {k: float(wordcount[k])/int(sum) for k in wordcount.keys()}

    words = {k: Word(k, wordcount[k], args.fh, args.fc) for k in wordcount.keys()}

    patterns = dict()

    with open(args.work_dir+"/train.data", encoding='ISO-8859-1') as input_file:
        train_docs = [line.rstrip().split() for line in input_file]

    with open(args.work_dir+"/dev.data", encoding='ISO-8859-1') as input_file:
        dev_docs = [line.rstrip().split() for line in input_file]

    train_labels = read_labels(args.work_dir+"/train.labels")
    dev_labels = read_labels(args.work_dir+"/dev.labels")

    for doc in train_docs:
        add_patterns(doc, words, patterns, args.max_pattern_len, args.use_CW_tokens)

    thr = args.min_pattern_frequency*len(train_docs)

    print("Read", len(patterns), "patterns")
    patterns = {k: patterns[k] for k in patterns.keys() if patterns[k] >= thr}

    s = 0
    for p in patterns.keys():
        p.set_freq(patterns[p])
        s += patterns[p]

    print("Read", len(patterns), "patterns", s)

    trie = build_trie(patterns)

    # print(trie)

    # sys.exit(-1)

    # print([x.__str__() for x in patterns if x.size() >= 3])

    train_features = lil_matrix((len(train_docs), len(patterns)), dtype=np.int8)
    dev_features = lil_matrix((len(dev_docs), len(patterns)))

    for (i, doc) in enumerate(train_docs):
        add_patterns(doc, words, patterns, args.max_pattern_len, args.use_CW_tokens, trie, train_features, i)

    for (i, doc) in enumerate(dev_docs):
        add_patterns(doc, words, patterns, args.max_pattern_len, args.use_CW_tokens, trie, dev_features, i)

    # print([x.__str__() for x in patterns.keys()])
    # print("df",dev_features)

    train(train_features, train_labels, dev_features, dev_labels)


    return 0

def train(train_features, train_labels, dev_features, dev_labels):
    max_r = -1
    argmax = None

    Cs = [1,0.5,0.1,0.05,0.01]

    for C in Cs:
        print("Testing", C)
        clf = linear_model.LogisticRegression(C=C)

        clf.fit(train_features, train_labels)

        train_predicted_labels = clf.predict(train_features)

        train_acc = evaluate(train_predicted_labels, train_labels)

        dev_predicted_labels = clf.predict(dev_features)
        dev_acc = evaluate(dev_predicted_labels, dev_labels)

        print("Train: {}, dev: {}".format(train_acc, dev_acc))

        if dev_acc > max_r:
            max_r = dev_acc
            argmax = C

    clf = linear_model.LogisticRegression(C=argmax)

    return clf



def evaluate(predicted, gold):
    return 1.*sum(predicted==gold)/len(predicted)


def build_trie(patterns):
    trie = dict()

    for (i, p) in enumerate(patterns):
        local_trie = trie
        for hfw in p.hfws:
            if hfw not in local_trie:
                local_trie[hfw] = dict()

            local_trie = local_trie[hfw]

        local_trie[INDEX_TOKEN] = i

    return trie


class StackElement():
    def __init__(self, pattern):
        self.pattern = pattern

    def new_element(self, pattern, other=None):
        raise NotImplementedError

    def finish(self, k1, k2, k3):
        raise NotImplementedError

class PlainStackElement(StackElement):
    def __init__(self, pattern, patterns):
        super(PlainStackElement, self).__init__(pattern)
        self.patterns = patterns

    def new_element(self, pattern, w=None):
        return PlainStackElement(pattern, self.patterns)

    def finish(self, k2, k3):
        # Updating new pattern count
        if self.pattern in self.patterns:
            self.patterns[self.pattern] += 1
        else:
            self.patterns[self.pattern] = 1


class TrieStackElement(StackElement):
    def __init__(self, pattern, trie, features):
        super(TrieStackElement, self).__init__(pattern)
        self.trie = trie
        self.features = features

    def new_element(self, pattern, w=None):
        trie = self.get_trie(w)

        if trie is not None:
            return TrieStackElement(pattern, trie, self.features)

    def get_trie(self, w):
        if w is None:
            return self.trie
        elif w in self.trie:
            return self.trie[w]
        else:
            return None

    def finish(self, w, index):
        trie = self.get_trie(w)

        if trie is not None and INDEX_TOKEN in trie:
            self.features[index, trie[INDEX_TOKEN]] = 1  # / patterns[p2obj]


def add_patterns(doc, words, patterns, max_size, use_CW_tokens, trie=None, features=None, index=None):
    # print("Trie is",trie.keys() if trie is not None else None)
    patterns_stack = []

    stackElement = PlainStackElement(None, patterns) if trie is None else TrieStackElement(None, trie, features)

    for w in doc:
        if w in words and words[w].senses[0] == 0:
            wobj = words[w]

            # Pattern object of w
            pobj = Pattern(w)

            new_stack = []

            # Traversing all patterns in our stack.
            for element in patterns_stack:
                # Pattern type mode: just one element, the pattern object.

                # new pattern object: pattern object + w
                p2obj = element.pattern.add_hfw(w)

                # if pattern has not reached maximum length, adding pattern to stack for the next word(s).
                new_element = element.new_element(p2obj, w)

                if new_element is not None:
                    if p2obj.size() < max_size:
                        new_stack.append(new_element)

                    new_element.finish(w, index)

                # If w can also be a CW, adding original pattern object again.
                if len(wobj.senses) == 2:
                    new_stack.append(element)

            patterns_stack = new_stack

            # w alone can also be a pattern
            new_element = stackElement.new_element(pobj, w)

            if new_element is not None:
                patterns_stack.append(new_element)

                new_element.finish(None, index)



class Pattern():
    def __init__(self, first_hfw):
        self.hfws = [first_hfw]

    def add_hfw(self, hfw):
        p = Pattern(self.hfws[0])

        for orig_hfw in self.hfws[1:]:
            p.hfws.append(orig_hfw)

        p.hfws.append(hfw)

        return p

    def set_freq(self, freq):
        self.freq = freq

    def score(self):
        return 1./self.freq

    def size(self):
        return len(self.hfws)

    def __str__(self):
        return " ".join([x for x in self.hfws])

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return (self.hfws == other.hfws)

    def __ne__(self, other):
        return not (self == other)


class Word():
    word_re = re.compile("^[a-z]+$", re.I)

    def __init__(self, word, frequency, fh, fc):
        self.senses = []

        self.word = word


        if Word.word_re.match(word):
            if frequency >= fh:
                # print(word, "is hfw")
                self.senses.append(0)

            if frequency <= fc:
                self.senses.append(1)
        else:
            self.senses.append(0)

    def __str__(self):
        return self.word

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return (self.word == other.word)

    def __ne__(self, other):
        return not (self == other)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=100)
    parser.add_argument("-d", "--work_dir", help="Work dir (where {train,dev}.{data,labels} are found", required=True)
    parser.add_argument("--fh", help="High frequency word minimal threshold", type=float, default=0.0001)
    parser.add_argument("--fc", help="Content word maximal threshold", type=float, default=0.001)
    parser.add_argument("-m", "--min_pattern_frequency", help="Minimal pattern frequency", type=float, default=0.005)
    parser.add_argument("-c", "--use_CW_tokens", help="Use CW tokens in pattern", action='store_true')
    parser.add_argument("-x", "--max_pattern_len", help="Maximum number of HFWs in pattern", type=int, default=6)

    main(parser.parse_args())

