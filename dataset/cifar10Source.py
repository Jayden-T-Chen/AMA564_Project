"""CIFAR-10 source data used to train the frozen backbone."""

import numpy as np
from torchvision.datasets import CIFAR10


def loadSplit(root, train, download=True):
    """Return (images uint8 [n,32,32,3], labels int64 [n])."""
    dataset = CIFAR10(root=root, train=train, download=download)
    return np.asarray(dataset.data), np.asarray(dataset.targets)


def trainingStatistics(root):
    """Channel mean and std on the 0-1 scale, fit on the training split only."""
    images, _ = loadSplit(root, train=True)
    channels = images.reshape(-1, 3).astype(np.float64) / 255.0
    return {
        "mean": channels.mean(axis=0).tolist(),
        "std": channels.std(axis=0).tolist(),
    }
