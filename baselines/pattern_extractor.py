#!/usr/bin/env python

import sys
import argparse
import re
from collections import Counter
import numpy as np
from data import read_labels
from sklearn import linear_model
from scipy.sparse import lil_matrix
from soft_patterns import CW_TOKEN
import pickle


INDEX_TOKEN = "###INDEX###"


def main(args):
    with open(args.work_dir+"/train.data", encoding="ISO-8859-1") as ifh:
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

    with open(args.work_dir+"/test.data", encoding='ISO-8859-1') as input_file:
        test_docs = [line.rstrip().split() for line in input_file]

    train_labels = read_labels(args.work_dir+"/train.labels")
    dev_labels = read_labels(args.work_dir+"/dev.labels")
    test_labels = read_labels(args.work_dir+"/test.labels")

    for doc in train_docs:
        add_patterns(doc, words, patterns, args.max_pattern_len, args.use_CW_tokens, args.min_pattern_length)
        # sys.exit(-1)

    if args.min_pattern_frequency < 1:
        thr = args.min_pattern_frequency*len(train_docs)
    else:
        thr = args.min_pattern_frequency

    print("Read", len(patterns), "patterns")
    patterns = {k: patterns[k] for k in patterns.keys() if patterns[k] >= thr}

    s = 0
    for p in patterns.keys():
        p.set_freq(patterns[p])
        s += patterns[p]

    pattern_keys = list(patterns.keys())

    print("Read", len(patterns), "patterns", s)

    trie = build_trie(pattern_keys)

    # print(trie)

    # sys.exit(-1)

    # print([x.__str__() for x in patterns if x.size() >= 3])

    train_features = lil_matrix((len(train_docs), len(patterns)), dtype=np.int8)
    dev_features = lil_matrix((len(dev_docs), len(patterns)))
    test_features = lil_matrix((len(test_docs), len(patterns)))

    for (i, doc) in enumerate(train_docs):
        add_patterns(doc, words, patterns, args.max_pattern_len, args.use_CW_tokens, args.min_pattern_length, trie, train_features, i)

    for (i, doc) in enumerate(dev_docs):
        add_patterns(doc, words, patterns, args.max_pattern_len, args.use_CW_tokens, args.min_pattern_length, trie, dev_features, i)

    for (i, doc) in enumerate(test_docs):
        add_patterns(doc, words, patterns, args.max_pattern_len, args.use_CW_tokens, args.min_pattern_length, trie, test_features, i)

    # print([x.__str__() for x in patterns.keys()])
    # print("df",dev_features)
    # print("tf", train_features)

    clf = train(train_features, train_labels, dev_features, dev_labels)

    gen_salient_patterns(train_features, clf, pattern_keys, args.n_salient_features)

    if args.model_ofile is not None:
        print("Saving best model to", args.model_ofile)
        pickle.dump(clf, open(args.model_ofile, 'wb'))

    test_predicted_labels = clf.predict(test_features)
    test_acc = evaluate(test_predicted_labels, test_labels)

    print("Test accuracy: {}".format(test_acc))

    return 0

def train(train_features, train_labels, dev_features, dev_labels):
    max_r = -1
    argmax = None

    Cs = [1,0.5,0.1,0.05,0.01, 0.005, 0.001]

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

    clf.fit(train_features, train_labels)

    print("Num of params = ", clf.coef_[0].shape)

    return clf



def evaluate(predicted, gold):
    return 1.*sum(predicted==gold)/len(predicted)


def gen_salient_patterns(train_features, clf, pattern_keys, n=5):
    weights = clf.coef_[0]

    sorted_w = np.argsort(weights)

    top_n = sorted_w[:n]
    last_n = list(reversed(sorted_w[-n:]))

    print(["{}, '{}': {:,.3f}".format(i, pattern_keys[i].__str__(), weights[i]) for i in top_n])
    print(["{},'{}': {:,.3f}".format(i, pattern_keys[i].__str__(), weights[i]) for i in last_n])


def build_trie(patterns):
    trie = dict()

    for (i, p) in enumerate(patterns):
        # print("patt",p,i)
        local_trie = trie
        for element in p.elements:
            if element not in local_trie:
                local_trie[element] = dict()

            local_trie = local_trie[element]

        local_trie[INDEX_TOKEN] = i

    return trie


class StackElement():
    def __init__(self, pattern, min_pattern_length):
        self.pattern = pattern
        self.min_pattern_length = min_pattern_length

    def new_element(self, pattern, other=None):
        raise NotImplementedError

    def finish(self, k1):
        return self.min_pattern_length <= self.pattern.n_hfws()


class PlainStackElement(StackElement):
    def __init__(self, pattern, min_pattern_length, patterns):
        super(PlainStackElement, self).__init__(pattern, min_pattern_length)
        self.patterns = patterns

    def new_element(self, pattern, w=None):
        return PlainStackElement(pattern, self.min_pattern_length, self.patterns)

    def finish(self, k1):
        if not super(PlainStackElement, self).finish(k1):
            return False

        # Updating new pattern count
        if self.pattern in self.patterns:
            self.patterns[self.pattern] += 1
        else:
            self.patterns[self.pattern] = 1


class TrieStackElement(StackElement):
    def __init__(self, pattern, min_pattern_length, trie, features):
        super(TrieStackElement, self).__init__(pattern, min_pattern_length)
        self.trie = trie
        self.features = features

    def new_element(self, pattern, w=None):
        trie = self.get_trie(w)

        return TrieStackElement(pattern, self.min_pattern_length, trie, self.features) if trie is not None else None

    def get_trie(self, w):
        if w is None:
            return self.trie
        elif w in self.trie:
            return self.trie[w]
        else:
            return None

    def finish(self, index):
        if not super(TrieStackElement, self).finish(index):
            return False

        if INDEX_TOKEN in self.trie:
            self.features[index, self.trie[INDEX_TOKEN]] = 1  # / patterns[p2obj]


def add_patterns(doc, words, patterns, max_size, use_CW_tokens, min_pattern_length, trie=None, features=None, index=None):
    # print("Trie is",trie.keys() if trie is not None else None)
    patterns_stack = []

    stackElement = PlainStackElement(None, min_pattern_length, patterns) if trie is None \
        else TrieStackElement(None, min_pattern_length, trie, features)

    # if trie is not None:
    #     print("new doc")
    for w in doc:
        new_stack = []

        # if trie is not None:
            # print(w)
        if w in words and words[w].senses[0] == 0:
            wobj = words[w]

            # Traversing all patterns in our stack.
            for element in patterns_stack:
                # Pattern type mode: just one element, the pattern object.

                # if trie is not None:
                #     print(element.trie)

                # new pattern object: pattern object + w
                p2obj = element.pattern.add_hfw(w)

                new_element = element.new_element(p2obj, w)

                if new_element is not None:
                    # if pattern has not reached maximum length, adding pattern to stack for the next word(s).
                    if p2obj.n_hfws() < max_size:
                        new_stack.append(new_element)

                    new_element.finish(index)

                # If w can also be a CW, adding original pattern object again.
                if len(wobj.senses) == 2:
                    if use_CW_tokens:
                        p3obj = element.pattern.add_cw()
                        new_element = element.new_element(p3obj, CW_TOKEN)

                        if new_element is not None:
                            new_stack.append(new_element)
                    else:
                        new_stack.append(element)

            # Pattern object of w
            pobj = Pattern(w)

            # w alone can also be a pattern
            new_element = stackElement.new_element(pobj, w)

            if new_element is not None:
                new_stack.append(new_element)

                new_element.finish(index)

        elif use_CW_tokens:
            for element in patterns_stack:
                # Pattern type mode: just one element, the pattern object.

                # new pattern object: pattern object + CW
                p2obj = element.pattern.add_cw()

                # if pattern has not reached maximum length, adding pattern to stack for the next word(s).
                new_element = element.new_element(p2obj, CW_TOKEN)

                if new_element is not None:
                    new_stack.append(new_element)
        else:
            continue

            # patterns_stack = new_stack

        patterns_stack = new_stack
        # print(len(patterns_stack))


class Pattern():
    def __init__(self, first_hfw):
        self.elements = [first_hfw]
        self._n_hfws = 1

    def clone(self):
        p = Pattern(self.elements[0])

        for orig_element in self.elements[1:]:
            p.elements.append(orig_element)

        p._n_hfws = self.n_hfws()

        return p

    def add_hfw(self, hfw):
        p = self.clone()

        p.elements.append(hfw)

        p._n_hfws += 1

        return p

    def add_cw(self):
        p = self.clone()

        p.elements.append(CW_TOKEN)

        return p

    def set_freq(self, freq):
        self.freq = freq

    def score(self):
        return 1./self.freq

    def n_hfws(self):
        return self._n_hfws

    def __str__(self):
        return " ".join([x for x in self.elements])

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return (self.elements == other.elements)

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
    parser.add_argument("--fc", help="Content word maximal threshold", type=float, default=0.01)
    parser.add_argument("-m", "--min_pattern_frequency", help="Minimal pattern frequency", type=float, default=0.001)
    parser.add_argument("-c", "--use_CW_tokens", help="Use CW tokens in pattern", action='store_true')
    parser.add_argument("-x", "--max_pattern_len", help="Maximum number of HFWs in pattern", type=int, default=6)
    parser.add_argument("-n", "--n_salient_features", help="Number of salient features to print", type=int, default=5)
    parser.add_argument("-i", "--min_pattern_length", help="Minimum number of HFWs in pattern", type=int, default=1)
    parser.add_argument("-o", "--model_ofile", help="Model output file")

    main(parser.parse_args())

