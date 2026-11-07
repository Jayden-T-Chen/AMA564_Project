"""configIO.setPath: the two-level override every ablation and ladder row uses."""

import unittest

from configIO import loadConfig, setPath


class SetPathTests(unittest.TestCase):
    def testWritesTheKeyAndReturnsTheSameDict(self):
        config = {"consistencyGate": {"weight": 4.0}}
        self.assertIs(setPath(config, "consistencyGate.weight", 16.0), config)
        self.assertEqual(config["consistencyGate"]["weight"], 16.0)

    def testLeavesTheOtherKeysOfTheRealConfigAlone(self):
        config = loadConfig("gate")
        floor = config["consistencyGate"]["weightFloor"]
        setPath(config, "entropyGate.keepFraction", 0.5)
        self.assertEqual(config["entropyGate"]["keepFraction"], 0.5)
        self.assertEqual(config["consistencyGate"]["weightFloor"], floor)

    def testUnknownSectionRaisesKeyError(self):
        with self.assertRaises(KeyError):
            setPath({"entropyGate": {}}, "missing.key", 1)

    def testThreeLevelKeyRaisesValueError(self):
        # setPath splits on one dot, and every override in configs/gate.yaml is
        # two-level, so a deeper key is a typo rather than a supported path.
        with self.assertRaises(ValueError):
            setPath({"a": {"b": {}}}, "a.b.c", 1)
