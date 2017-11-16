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
        add_patterns(doc, words, patterns, args.max_pattern_len)

    thr = args.min_pattern_frequency*len(train_docs)

    print("Read", len(patterns), "patterns")
    patterns = {k: patterns[k] for k in patterns.keys() if patterns[k] >= thr}

    for p in patterns.keys():
        p.set_freq(patterns[p])

    print("Read", len(patterns), "patterns")

    trie = build_trie(patterns)

    # print(trie)

    # sys.exit(-1)

    # print([x.__str__() for x in patterns if x.size() >= 3])

    train_features = lil_matrix((len(train_docs), len(patterns)), dtype=np.int8)
    dev_features = lil_matrix((len(dev_docs), len(patterns)))

    for (i, doc) in enumerate(train_docs):
        add_patterns(doc, words, patterns, args.max_pattern_len, trie, train_features, i)

    for (i, doc) in enumerate(dev_docs):
        add_patterns(doc, words, patterns, args.max_pattern_len, trie, dev_features, i)

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

def add_patterns(doc, words, patterns, max_size, trie=None, features=None, index=None):
    # print("Trie is",trie.keys() if trie is not None else None)
    patterns_stack = []

    local_trie = None

    for w in doc:
        if w not in words:
            # print("not found")
            continue

        # if trie is not None:
        #     print(w)

        wobj = words[w]

        # HFW
        if wobj.senses[0] == 0:
            # Pattern object of w
            pobj = Pattern(w)

            new_stack = []

            # Traversing all patterns in our stack.
            for e in patterns_stack:
                # Pattern type mode: just one element, the pattern object.
                if trie is None:
                    p2 = e
                # Document scoring more: two elements, pattern object and local trie
                else:
                    p2 = e[0]
                    local_trie = e[1]

                # new pattern object: pattern object + w
                p2obj = p2.add_hfw(w)

                # if pattern has not reached maximum length, adding pattern to stack for the next word(s).
                if p2obj.size() < max_size:
                    if trie is None:
                        new_stack.append(p2obj)
                    elif w in local_trie:
                        new_stack.append([p2obj, local_trie[w]])

                if trie is None:
                    # Updating new pattern count
                    if p2obj in patterns:
                        patterns[p2obj] += 1
                    else:
                        patterns[p2obj] = 1

                    # If w can also be a CW, adding original pattern object again.
                    if len(wobj.senses) == 2:
                        new_stack.append(pobj)
                else:
                    if w in local_trie:
                        if INDEX_TOKEN in local_trie[w]:
                            features[index, local_trie[w][INDEX_TOKEN]] = 1# / patterns[p2obj]
                            # if p2obj in patterns:
                            #     print("Adding", 1/patterns[p2obj],"to", index, local_trie[w][INDEX_TOKEN], p2obj)
                            #     features[index, local_trie[w][INDEX_TOKEN]] = 1/patterns[p2obj]
                            # else:
                            #     print(p2obj.__str__(), "not found:(")

                        # If w can also be a CW, adding original pattern object again.
                        if len(wobj.senses) == 2:
                            new_stack.append([pobj, local_trie])

            patterns_stack = new_stack

            # w alone can also be a pattern
            if trie is None:
                patterns_stack.append(pobj)
                if pobj in patterns:
                    patterns[pobj] += 1
                else:
                    patterns[pobj] = 1
            elif w in trie:
                patterns_stack.append([pobj, trie[w]])
                if INDEX_TOKEN in trie[w]:
                    features[index, trie[w][INDEX_TOKEN]] = 1
                    # if pobj in patterns:
                    #     print("Adding", 1 / patterns[pobj], "to", index)
                    #     features[index, trie[w][INDEX_TOKEN]] = 1/patterns[pobj]
                    # else:
                    #     print(pobj.__str__(),"not found:(")



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
    parser.add_argument("-x", "--max_pattern_len", help="Maximum number of HFWs in pattern", type=int, default=6)

    main(parser.parse_args())

