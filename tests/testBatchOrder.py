"""batchOrder: the shared permutation and the head padding that fills the last batch.

The two padded lengths asserted here are the ones README 2.3 quotes for the
validation and test cells.
"""

import unittest

import numpy as np

from run.runUtils import batchOrder


class BatchOrderTests(unittest.TestCase):
    def testLengthIsAWholeNumberOfBatches(self):
        for count, batchSize in [(5000, 64), (10000, 64), (100, 16), (7, 16), (64, 64)]:
            order = batchOrder(count, batchSize, 7)
            self.assertEqual(len(order) % batchSize, 0)
            self.assertGreaterEqual(len(order), count)

    def testValidationAndTestCellLengths(self):
        self.assertEqual(len(batchOrder(5000, 64, 7)), 5056)
        self.assertEqual(len(batchOrder(10000, 64, 7)), 10048)

    def testTheFirstCountIndicesAreAPermutation(self):
        order = batchOrder(5000, 64, 7)
        self.assertEqual(sorted(order[:5000].tolist()), list(range(5000)))

    def testPaddingRepeatsTheHeadOfThePermutation(self):
        order = batchOrder(5000, 64, 7)
        np.testing.assert_array_equal(order[5000:], order[:len(order) - 5000])

    def testNoPaddingWhenTheCountDivides(self):
        self.assertEqual(len(batchOrder(64, 64, 7)), 64)

    def testCountSmallerThanOneBatchIsPaddedToThatBatch(self):
        order = batchOrder(7, 16, 7)
        self.assertEqual(len(order), 16)
        self.assertEqual(sorted(order[:7].tolist()), list(range(7)))

    def testSameSeedGivesTheSameOrderAndAnotherSeedDoesNot(self):
        np.testing.assert_array_equal(batchOrder(500, 64, 7), batchOrder(500, 64, 7))
        self.assertFalse(np.array_equal(batchOrder(500, 64, 7), batchOrder(500, 64, 8)))
