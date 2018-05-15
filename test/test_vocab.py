#!/usr/bin/env python

import unittest

from data import Vocab, read_embeddings, UNK_TOKEN

EMBEDDINGS_FILENAME = 'test/data/glove.6B.50d.20words.txt'


class TestVocab(unittest.TestCase):
    def test_read_embeddings(self):
        """ Tests that `data.read_embeddings` works for a small file """
        max_vocab_size = 4
        vocab, vecs, dim = read_embeddings(EMBEDDINGS_FILENAME, max_vocab_size=max_vocab_size)
        self.assertEqual(len(vocab), max_vocab_size+2)
        self.assertEqual(dim, 50)
        self.assertEqual(vocab(UNK_TOKEN), 0)
        self.assertEqual(vocab("the"), 3)
        self.assertEqual(vocab(","), 4)
        self.assertAlmostEqualList(
            list(vecs[vocab("the")])[:10],
            [
                0.0841414179717338836,
                0.050259400093738082,
                -0.08301819043038873,
                0.024497632935789507,
                0.069501213835168801,
                -0.0089489832984913243,
                -0.10001958794687831,
                -0.035955359038543321,
                -0.00013290116839109539,
                -0.13217046660344609
            ],
            7
        )
        self.assertAlmostEqualList(
            list(vecs[vocab(",")])[:10],
            [
                0.0030016593815816841,
                0.052886911297237889,
                -0.037739038679673299,
                0.091452238178075698,
                0.14250568295327015,
                0.10654428051177782,
                -0.095697572962977706,
                -0.12425811297566139,
                -0.081288893303752177,
                -0.05345861340398955
            ]
        , 7)

    def test_denumberize_numberize(self):
        """ Tests that `denumberize` is left inverse of `numberize` """
        fixture1 = [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["a", "f", "b"],
            ["b", "e", "d"]
        ]
        fixture2 = [
            [0, 1, 2],
            [3, 4, 5],
            [0, 5, 1],
            [2, 4, 3]
        ]

        for fixture in (fixture1, fixture2):
            v = Vocab.from_docs(fixture)
            for doc in fixture:
                self.assertEqual(v.denumberize(v.numberize(doc)), doc)

    def assertAlmostEqualList(self,
                              first,
                              second,
                              places=None,
                              msg=None,
                              delta=None):
        for i, j in zip(first, second):
            self.assertAlmostEqual(i, j, places, msg, delta)


if __name__ == "__main__":
    unittest.main()
