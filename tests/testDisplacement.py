"""protocols.displacement: the distance one cell actually moves the trainable parameters."""

import unittest

import torch

from evaluation.protocols import displacement, snapshot, trainableParameters


class TwoParameters(torch.nn.Module):
    """One trainable parameter and one frozen, so the exclusion is observable."""

    def __init__(self):
        super().__init__()
        self.moved = torch.nn.Parameter(torch.zeros(3))
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)


class DisplacementTests(unittest.TestCase):
    def setUp(self):
        self.model = TwoParameters()

    def testOnlyTrainableParametersAreListed(self):
        self.assertEqual(len(trainableParameters(self.model)), 1)

    def testNoTrainableParametersGivesZero(self):
        self.model.moved.requires_grad_(False)
        self.assertEqual(displacement(snapshot(self.model), self.model), 0.0)

    def testAFrozenParameterMovingDoesNotCount(self):
        before = snapshot(self.model)
        with torch.no_grad():
            self.model.frozen += 100.0
        self.assertEqual(displacement(before, self.model), 0.0)

    def testAKnownMoveGivesItsL2Norm(self):
        before = snapshot(self.model)
        with torch.no_grad():
            self.model.moved += torch.tensor([3.0, 4.0, 0.0])
        self.assertAlmostEqual(displacement(before, self.model), 5.0, places=5)

    def testAnUnchangedModelGivesZero(self):
        self.assertEqual(displacement(snapshot(self.model), self.model), 0.0)

    def testSnapshotIsACopyTakenAtThatMoment(self):
        before = snapshot(self.model)
        with torch.no_grad():
            self.model.moved += 1.0
        self.assertTrue(torch.equal(before[0], torch.zeros(3)))
