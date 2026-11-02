"""Online mixed stream: several corruptions interleaved into one sequence.

Batches cross corruption boundaries, which is the point of the setting: the BN
statistics a method sees come from a mixture of drift types.
"""

import numpy as np

from dataset.cifar10CLoader import cell


def buildStream(root, corruptions, severity, imagesPerCorruption):
    """Return (images, labels, sourceId) for the interleaved stream."""
    blocks = [cell(root, corruption, severity, imagesPerCorruption) for corruption in corruptions]
    images = np.concatenate([block[0] for block in blocks])
    labels = np.concatenate([block[1] for block in blocks])
    sourceId = np.concatenate([
        np.full(len(block[1]), index) for index, block in enumerate(blocks)
    ])
    return images, labels, sourceId
