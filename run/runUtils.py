"""Shared helpers for the run scripts: device, seeding, CSV records."""

import csv
import random
from pathlib import Path

import numpy as np
import torch

from configIO import loadConfig


def resolveDevice(order=None):
    config = order or loadConfig("base")["deviceOrder"]
    for name in config:
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batchOrder(count, batchSize, seed):
    """Permutation of `count` indices, padded to a whole number of batches.

    Every method at the same seed sees exactly the same batches, which is what
    makes the online comparison fair.
    """
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(count)
    remainder = -len(permutation) % batchSize
    permutation = np.concatenate([permutation, permutation[:remainder]])
    return permutation


def appendRows(csvPath, rows, fieldNames):
    path = Path(csvPath)
    path.parent.mkdir(parents=True, exist_ok=True)
    writeHeader = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldNames)
        if writeHeader:
            writer.writeheader()
        writer.writerows(rows)


def readRows(csvPath):
    with open(csvPath, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
