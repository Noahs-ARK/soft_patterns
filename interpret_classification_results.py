#!/usr/bin/env python3
"""
Script to visualize the patterns in a SoftPatterns model based on their
highest-scoring spans in the dev set.
"""
import argparse
from collections import OrderedDict
import sys
from soft_patterns import MaxPlusSemiring, Batch, argmax, SoftPatternClassifier, ProbSemiring, \
    LogSpaceMaxTimesSemiring, soft_pattern_arg_parser, general_arg_parser
import torch
from torch.nn import LSTM

from rnn import Rnn
from visualize_efficiently import get_top_scoring_spans_for_doc
from torch.nn.functional import softmax
from torch.autograd import Variable
from data import vocab_from_text, read_embeddings, read_docs, read_labels
from util import chunked
import numpy as np

SCORE_IDX = 0
START_IDX_IDX = 1
END_IDX_IDX = 2


def interpret_documents(model, batch_size, dev_data, dev_text, ofile, max_doc_len):
    j = 0
    with open(ofile, "w") as ofh:
        for batch_idx, chunk in enumerate(chunked(dev_data, batch_size)):
            batch = Batch([x for x, y in chunk], model.embeddings, model.to_cuda)
            res, scores = model.forward(batch, 1)
            print("ss", scores.size())

            output = softmax(res).data

            predictions = [int(x) for x in argmax(output)]

            num_patts = scores.size()[1]

            diffs = np.zeros((num_patts, batch.size()))

            # Traversing all patterns.
            for i in range(num_patts):
                # Copying scores data to numpy array.
                scores_data = np.array(scores.data.numpy(), copy=True)

                # Zeroing out pattern number i across batch
                scores_data[:, i] = 0

                # Running mlp.forward() with zeroed out scores.
                forwarded = softmax(model.mlp.forward(Variable(torch.FloatTensor(scores_data)))).data.numpy()

                # Computing difference between forwarded scores and original scores.
                for k in range(batch.size()):
                    # diffs[i,k] = output[k, predictions[k]] - \
                    #              output[k, 1 - predictions[k]] - \
                    #              forwarded[k, predictions[k]] + \
                    #              forwarded[k, 1 - predictions[k]]

                    diffs[i, k] = forwarded[k, 1 - predictions[k]] - output[k, 1 - predictions[k]]

            # Now, traversing documents in batch
            for i in range(batch.size()):
                # Document string
                text_str = str(" ".join(dev_text[j]).encode('utf-8'))[2:-1]

                # Top ten patterns with largest differences between leave-one-out score and original score.
                top_ten_deltas = sorted(enumerate(diffs[:, i]), key=lambda x: x[1], reverse=True)[:10]
                top_ten_neg_deltas = sorted(enumerate(diffs[:, i]), key=lambda x: x[1])[:10]
                # Top ten patterns with largest overall score (regardless of classification)
                top_ten_scores = sorted(enumerate(scores.data.numpy()[i, :]), key=lambda x: x[1], reverse=True)[:10]

                top_scoring_spans = get_top_scoring_spans_for_doc(model, dev_data[j], max_doc_len)

                # Printing out everything.
                ofh.write("{}   {}   {} All in, predicted: {:>2,.3f}   All in, not-predicted: {:>2,.3f}    Leave one out: +res: {} -res: {} Patt scores: {}\n".format(
                                                        dev_data[j][1],
                                                        predictions[i],
                                                        text_str,
                                                        output[i, predictions[i]],
                                                        output[i, 1-predictions[i]],
                                                        " ".join(["{}:{:>2,.3f}".format(i,x) for (i,x) in top_ten_deltas]),
                                                        " ".join(["{}:{:>2,.3f}".format(i,x) for (i,x) in top_ten_neg_deltas]),
                                                        " ".join(["{}:{:>2,.3f}".format(i, x) for (i, x) in
                                                                                           top_ten_scores])))
                ofh.write("Top ten deltas:\n")
                for l in top_ten_deltas:
                    s = top_scoring_spans[l[0]].display(dev_text[j])
                    ofh.write(str(int(l[0]))+" "+str(s.encode('utf-8'))[2:-1]+"\n")

                ofh.write("Top ten negative deltas:\n")
                for l in top_ten_neg_deltas:
                    s = top_scoring_spans[l[0]].display(dev_text[j])
                    ofh.write(str(int(l[0]))+" "+str(s.encode('utf-8'))[2:-1]+"\n")
                j += 1


# TODO: refactor duplicate code with soft_patterns.py
def main(args):
    print(args)

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    pattern_specs = OrderedDict(sorted(([int(y) for y in x.split("-")] for x in args.patterns.split("_")),
                                       key=lambda t: t[0]))

    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab size:", len(dev_vocab))

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    num_padding_tokens = max(list(pattern_specs.keys())) - 1

    dev_input, dev_text = read_docs(args.vd, vocab, num_padding_tokens=num_padding_tokens)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))
    if n is not None:
        dev_data = dev_data[:n]

    num_classes = len(set(dev_labels))
    print("num_classes:", num_classes)

    semiring = \
        MaxPlusSemiring if args.maxplus else (
            LogSpaceMaxTimesSemiring if args.maxtimes else ProbSemiring
        )

    if args.use_rnn:
        rnn = Rnn(word_dim,
                  args.hidden_dim,
                  cell_type=LSTM,
                  gpu=args.gpu)
    else:
        rnn = None

    model = SoftPatternClassifier(pattern_specs, mlp_hidden_dim, num_mlp_layers, num_classes, embeddings, vocab,
                                  semiring, args.bias_scale_param, args.gpu, rnn=rnn, pre_computed_patterns=None)

    if args.gpu:
        print("Cuda!")
        model.to_cuda(model)
        state_dict = torch.load(args.input_model)
    else:
        state_dict = torch.load(args.input_model, map_location=lambda storage, loc: storage)

    # Loading model
    model.load_state_dict(state_dict)

    interpret_documents(model, args.batch_size, dev_data, dev_text, args.ofile, args.max_doc_len)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[soft_pattern_arg_parser(), general_arg_parser()])
    parser.add_argument("--ofile", help="Output file", required=True)

    sys.exit(main(parser.parse_args()))
