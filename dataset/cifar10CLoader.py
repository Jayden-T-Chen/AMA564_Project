"""Memory-mapped access to CIFAR-10-C.

Each corruption is one (50000, 32, 32, 3) uint8 array: five severity blocks of
10000 images, severity 1 first. labels.npy is the CIFAR-10 test labels tiled
the same way, so a row index works for both files.
"""

import numpy as np

IMAGES_PER_SEVERITY = 10000


def blockSlice(severity, count=None):
    """Row range of one severity block."""
    start = (severity - 1) * IMAGES_PER_SEVERITY
    return slice(start, start + (IMAGES_PER_SEVERITY if count is None else count))


def cell(root, corruption, severity, count=None):
    """Return (images uint8 [n,32,32,3], labels uint8 [n]) for one cell."""
    images = np.load(f"{root}/{corruption}.npy", mmap_mode="r")[blockSlice(severity, count)]
    labels = np.load(f"{root}/labels.npy", mmap_mode="r")[blockSlice(severity, count)]
    return np.asarray(images), np.asarray(labels)
