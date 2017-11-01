#!/usr/bin/env python

import unittest

from data import Vocab


class TestVocab(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
