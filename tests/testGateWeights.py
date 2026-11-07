"""The gate equivalences the report quotes, and how batch-relative the weights are.

Two claims in README 3.2 are pinned here: with both `use` switches off the weights
are all ones, and with the entropy gate on at keepFraction 1.0 the quantile cutoff
still leaves the whole batch. The last test measures what the per-batch min/max
normalisation does to the same image at three batch sizes; README 2.3 quotes the
magnitude it prints.
"""

import unittest

import torch

from configIO import loadConfig, setPath
from methods.gatedTent import normaliseDisagreement, sampleWeights
from methods.losses import symmetricKl


def gateWith(overrides):
    """The frozen gate config with a list of (dotted key, value) overrides applied."""
    gate = loadConfig("gate")
    for path, value in overrides:
        setPath(gate, path, value)
    return gate


WEIGHTS_ON = [
    ("entropyGate.use", False),
    ("consistencyGate.use", True),
    ("consistencyGate.inWeights", True),
    ("consistencyGate.weightFloor", 0.2),
]


class GateWeightTests(unittest.TestCase):
    def testBothGatesOffGivesOnes(self):
        gate = gateWith([("entropyGate.use", False), ("consistencyGate.use", False)])
        weights = sampleWeights(torch.tensor([0.1, 0.9, 0.4, 2.0]), torch.zeros(4), gate)
        self.assertTrue(torch.equal(weights, torch.ones(4)))

    def testKeepFractionOneLeavesTheWholeBatch(self):
        gate = gateWith([
            ("entropyGate.use", True),
            ("entropyGate.keepFraction", 1.0),
            ("consistencyGate.use", False),
        ])
        weights = sampleWeights(torch.tensor([0.1, 0.9, 0.4, 2.0]), None, gate)
        self.assertTrue(torch.equal(weights, torch.ones(4)))

    def testKeepFractionHalfKeepsTheLowestEntropyHalf(self):
        gate = gateWith([
            ("entropyGate.use", True),
            ("entropyGate.keepFraction", 0.5),
            ("consistencyGate.use", False),
        ])
        weights = sampleWeights(torch.tensor([0.1, 0.9, 0.4, 2.0]), None, gate)
        self.assertTrue(torch.equal(weights, torch.tensor([1.0, 0.0, 1.0, 0.0])))

    def testWeightsAreDetached(self):
        entropies = torch.tensor([0.1, 0.9], requires_grad=True)
        weights = sampleWeights(entropies, torch.tensor([0.0, 1.0]), gateWith(WEIGHTS_ON))
        self.assertFalse(weights.requires_grad)

    def testTheFloorIsTheSmallestWeight(self):
        weights = sampleWeights(torch.zeros(3), torch.tensor([0.0, 0.5, 1.0]), gateWith(WEIGHTS_ON))
        torch.testing.assert_close(weights, torch.tensor([1.0, 0.5, 0.2]))

    def testNormalisationSpansZeroToOne(self):
        normalised = normaliseDisagreement(torch.tensor([2.0, 3.0, 6.0]))
        torch.testing.assert_close(normalised, torch.tensor([0.0, 0.25, 1.0]))

    def testAConstantBatchNormalisesToZero(self):
        normalised = normaliseDisagreement(torch.full((4,), 0.7))
        self.assertTrue(torch.equal(normalised, torch.zeros(4)))

    def testIdenticalViewsDisagreeByZero(self):
        logits = torch.tensor([[2.0, 0.1, -1.0], [0.0, 0.0, 3.0]])
        self.assertTrue(torch.equal(symmetricKl(logits, logits), torch.zeros(2)))

    def testTheSameImageIsWeightedDifferentlyInAnotherBatch(self):
        torch.manual_seed(7)
        disagreement = torch.rand(256)
        gate = gateWith(WEIGHTS_ON)
        bySize = {}
        for size in (16, 64, 256):
            batches = [sampleWeights(torch.zeros(size), disagreement[start:start + size], gate)
                       for start in range(0, 256, size)]
            bySize[size] = torch.cat(batches)
        gaps = {size: float((bySize[size] - bySize[256]).abs().max()) for size in (16, 64)}
        print(f"  batch-relative weights over one fixed 256-sample vector: "
              f"max |w16 - w256| {gaps[16]:.3f}, max |w64 - w256| {gaps[64]:.3f}")
        self.assertGreater(gaps[16], 0.0)
        self.assertGreater(gaps[64], 0.0)
