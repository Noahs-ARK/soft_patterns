#!/usr/bin/env python
"""
For visualizing soft patterns
"""
import argparse

import torch
from torch.autograd import Variable
from soft_patterns import MaxPlusSemiring, fixed_var, Batch, argmax
from util import chunked


def get_nearest_neighbors(w, embeddings):
    dot_products = torch.mm(w, embeddings[:1000, :])
    return argmax(dot_products)


def visualize_pattern(model, batch_size, dev_set=None, dev_text=None, n_top_scoring=5):
    nearest_neighbors = get_nearest_neighbors(model.diags.data, torch.FloatTensor(model.embeddings).t())

    if dev_set is not None:
        scores = get_top_scoring_sequences(model, dev_set, batch_size)

    start = 0
    for i, (pattern_length, num_patterns) in enumerate(model.pattern_specs.items()):
        # 1 above main diagonal
        viewed_tensor = \
            model.diags[model.starts[i]:model.ends[i], :].view(
                num_patterns,
                model.num_diags,
                pattern_length,
                model.word_dim
            )[:, 1, :-1, :]
        norms = torch.norm(viewed_tensor, 2, 2)
        viewed_biases = \
            model.bias[model.starts[i]:model.ends[i], :].view(
                num_patterns,
                model.num_diags,
                pattern_length
            )[:, 1, :-1]
        reviewed_nearest_neighbors = \
            nearest_neighbors[model.starts[i]:model.ends[i]].view(
                num_patterns,
                model.num_diags,
                pattern_length
            )[:, 1, :-1]

        if dev_set is not None:
            for p in range(num_patterns):
                patt_scores = scores[start + p, :, :]
                last_n = len(patt_scores) - n_top_scoring
                sorted_keys = sorted(range(len(patt_scores)), key=lambda i: patt_scores[i][0].data[0])

                print("Top scoring",
                      [(" ".join(dev_text[k][int(patt_scores[k][1].data[0]):int(patt_scores[k][2].data[0])]),
                        round(patt_scores[k][0].data[0], 3)) for k in sorted_keys[last_n:]],
                      "norms", [round(x, 3) for x in norms.data[p, :]],
                      'biases', [round(x, 3) for x in viewed_biases.data[p, :]],
                      'nearest neighbors', [model.vocab[x] for x in reviewed_nearest_neighbors[p, :]])
            start += num_patterns


def get_top_scoring_sequences(model, dev_set, batch_size):
    """
    Get top scoring sequence in doc for this pattern (for interpretation purposes)
    """
    n = 3  # max_score, start_idx, end_idx

    max_scores = Variable(MaxPlusSemiring.zero(model.total_num_patterns, len(dev_set), n))

    zero_paddings = [
        model.to_cuda(fixed_var(model.semiring.zero(num_patterns, 1)))
        for num_patterns in model.pattern_specs.values()
    ]

    debug_print = int(100 / batch_size) + 1
    eps_value = model.get_eps_value()
    self_loop_scale = model.self_loop_scale

    i = 0
    for batch_idx, batch in enumerate(chunked(dev_set, batch_size)):
        if i % debug_print == (debug_print - 1):
            print(".", end="", flush=True)
        i += 1
        batch_obj = Batch([x for x, y in batch], model.embeddings, model.to_cuda)

        transition_matrices = model.get_transition_matrices(batch_obj)

        for d, doc in enumerate(batch_obj.docs):
            doc_idx = batch_idx * batch_size + d
            for i in range(len(doc)):
                start = 0
                for k, (pattern_length, num_patterns) in enumerate(model.pattern_specs.items()):
                    hiddens = model.to_cuda(Variable(model.semiring.zero(num_patterns, pattern_length)))

                    # Start state
                    hiddens[:, 0] = model.to_cuda(model.semiring.one(num_patterns, 1))

                    for j in range(i, min(i + pattern_length - 1, len(doc))):
                        transition_matrix_val = transition_matrices[d][j][k]
                        hiddens = model.transition_once(
                            eps_value,
                            hiddens,
                            transition_matrix_val,
                            zero_paddings[k],
                            zero_paddings[k])

                        scores = hiddens[:, -1]

                        for p in range(num_patterns):
                            pattern_idx = start + p
                            if scores[p].data[0] > max_scores[pattern_idx, doc_idx, 0].data[0]:
                                max_scores[pattern_idx, doc_idx, 0] = scores[p]
                                max_scores[pattern_idx, doc_idx, 1] = i
                                max_scores[pattern_idx, doc_idx, 2] = j + 1
                    start += num_patterns
    print()
    return max_scores


def main(args):
    print(args)
    pattern_specs = OrderedDict([int(y) for y in x.split(":")] for x in args.patterns.split(","))
    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab:", len(dev_vocab))
    if args.td is not None:
        train_vocab = vocab_from_text(args.td)
        print("Train vocab:", len(train_vocab))
        dev_vocab |= train_vocab

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    dev_input, dev_text = read_docs(args.vd, vocab)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))

    if args.td is not None:
        if args.tl is None:
            print("Both training data (--td) and training labels (--tl) required in training mode")
            return -1

        np.random.shuffle(dev_data)
        num_iterations = args.num_iterations

        train_input, _ = read_docs(args.td, vocab)
        train_labels = read_labels(args.tl)

        print("training instances:", len(train_input))

        num_classes = len(set(train_labels))

        # truncate data (to debug faster)
        train_data = list(zip(train_input, train_labels))
        np.random.shuffle(train_data)
    else:
        num_classes = len(set(dev_labels))

    print("num_classes:", num_classes)

    if n is not None:
        if args.td is not None:
            train_data = train_data[:n]

        dev_data = dev_data[:n]

    semiring = MaxPlusSemiring if args.maxplus else ProbSemiring

    model = SoftPatternClassifier(pattern_specs,
                                  mlp_hidden_dim,
                                  num_mlp_layers,
                                  num_classes,
                                  embeddings,
                                  vocab,
                                  semiring,
                                  args.gpu,
                                  args.legacy)

    if args.gpu:
        model.to_cuda()

    model_file_prefix = 'model'
    # Loading model
    if args.input_model is not None:
        state_dict = torch.load(args.input_model)
        model.load_state_dict(state_dict)
        model_file_prefix = 'model_retrained'

    visualize_pattern(model, args.batch_size, dev_data, dev_text)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)
    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=100)
    parser.add_argument("-i", "--num_iterations", help="Number of iterations", type=int, default=10)
    parser.add_argument("-p", "--patterns",
                        help="Pattern lengths and numbers: a comma separated list of length:number pairs",
                        default="5:50,4:50,3:50,2:50")
    parser.add_argument("-d", "--mlp_hidden_dim", help="MLP hidden dimension", type=int, default=10)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=1)
    parser.add_argument("-y", "--num_mlp_layers", help="Number of MLP layers", type=int, default=2)
    parser.add_argument("-n", "--num_train_instances", help="Number of training instances", type=int, default=None)
    parser.add_argument("-m", "--model_save_dir", help="where to save the trained model")
    parser.add_argument("--input_model", help="Input model (to run test and not train)")
    parser.add_argument("--td", help="Train data file")
    parser.add_argument("--tl", help="Train labels file")
    parser.add_argument("--vd", help="Validation data file", required=True)
    parser.add_argument("--vl", help="Validation labels file", required=True)
    parser.add_argument("-l", "--learning_rate", help="Adam Learning rate", type=float, default=1e-3)
    parser.add_argument("--clip", help="Gradient clipping", type=float, default=None)
    parser.add_argument("--debug", help="Debug", type=int, default=0)
    parser.add_argument("--maxplus",
                        help="Use max-plus semiring instead of plus-times",
                        default=False, action='store_true')

    sys.exit(main(parser.parse_args()))
