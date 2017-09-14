#!/usr/bin/env python

import sys
import torch
import torch.nn
import numpy as np

def main(args):


def score_one_sentence(sentence, w, pi, eta, pattern_length, sigmoid):
    hidden = Variable(pi)

    s = 0
    for x in sentence:
        delta = compute_delta(x, w, pattern_length, sigmoid)
        hidden += torch.mm(hidden, delta) + pi
        s += torch.dot(hidden, eta)

    return s

def compute_delta(x, w, pattern_length, sigmoid):
    delta = Variable(torch.zeros(pattern_length, pattern_length))

    for i in range(pattern_length):
        for j in range(i, min(i+2, pattern_length-1)):
            delta[i][j] = sigmoid(torch.dot(w[i][j], x) - torch.log(torch.norm(w[i][j])))

    return delta


def learn_one_sentence(sentence, gold_output, w, pi, eta, pattern_length, sigmoid, mlp, optimizer):
    optimizer.zero_grad()
    score = score_one_sentence(sentence, w, pi, eta, pattern_length, sigmoid)

    output = mlp(score)

    softmax = nn.LogSoftmax()
    criterion = nn.NLLLoss()
    softmax_val = softmax(output)
    loss = criterion(softmax_val, gold_output)

    loss.backward()

    optimizer.step()

def learn_all_sentences(sentences, gold_outputs, pattern_length, word_dim, n_iterations):
    optimizer = torch.optim.Adam([var1, var2], lr=0.0001)
    sigmoid = nn.Sigmoid()
    mlp = None

    w = Variable(torch.normal(torch.zeros(pattern_length, pattern_length, word_dim), 1))

    pi = Variable(torch.zeros(pattern_length))
    pi[0] = 1
    eta = Variable(torch.zeros(pattern_length))
    eta[-1] = 1

    indices = range(sentences.size()[0])

    for i in n_iterations:
        np.random.shuffle(indices)

        for i in indices:
            sentence = sentences[i]
            gold = gold_outputs[i]
            learn_one_sentence(sentence, gold, w, pi, eta, pattern_length, sigmoid, mlp, optimizer)