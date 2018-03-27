#!/usr/bin/env python

import sys
import numpy as np
from itertools import compress


def main(args):
        argc = len(args)

        ratio = 0.05

        if argc < 3:
                print("Usage:", args[0], "<if> <of> <ratio={}>".format(ratio))
                return -1
        elif argc > 3:
            ratio = float(args[3])

        with open(args[1], encoding='utf-8') as ifh, open(args[2], 'w', encoding='utf-8') as ofh:
            for l in ifh:
                words = l.rstrip().split()
                indices = np.random.random_sample((len(words),)) > ratio
                selected_words = list(compress(words, indices))
                ofh.write(" ".join(selected_words)+"\n")

        return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
