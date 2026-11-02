"""The two batch loops every evaluation shares: one static cell, one mixed stream.

A method only has to provide predict(batch, targets) -> logits, which is why the
baselines and TENT sit in exactly the same loop. The loop hands the labels over
because the oracle reference point needs them; every other method ignores them,
and that is the whole difference between a method and an oracle.
"""

import torch

from dataset.cifar10CLoader import cell
from dataset.onlineStream import buildStream
from dataset.preprocess import prepare
from methods.losses import predictiveEntropy
from run.runUtils import batchOrder


def cellBatches(root, corruption, severity, count, meanStd, device, batchSize, seed):
    """One cell as full batches, in an order fixed by the seed and not by the method."""
    images, labels = cell(root, corruption, severity, count)
    block = prepare(images, meanStd, device)
    targets = torch.as_tensor(labels.astype("int64"), device=device)
    order = torch.as_tensor(batchOrder(len(labels), batchSize, seed), device=device)
    for start in range(0, len(order), batchSize):
        index = order[start:start + batchSize]
        yield block[index], targets[index]


def streamBatches(root, stream, meanStd, device, seed, batchSize):
    """The mixed stream as batches that cross corruption boundaries.

    Yields (batch, targets, sourceIds) so the caller can score each segment;
    sourceIds index configs/splits.yaml onlineStream.corruptions. batchSize is
    passed in rather than read from the section so the same stream can drive the
    batch-size sweep.
    """
    images, labels, sourceId = buildStream(
        root, stream["corruptions"], stream["severity"], stream["imagesPerCorruption"])
    block = prepare(images, meanStd, device)
    targets = torch.as_tensor(labels.astype("int64"), device=device)
    origin = torch.as_tensor(sourceId, device=device)
    order = torch.as_tensor(batchOrder(len(labels), batchSize, seed), device=device)
    for start in range(0, len(order), batchSize):
        index = order[start:start + batchSize]
        yield block[index], targets[index], origin[index]


def scoreBatch(logits, targets):
    """Per-image correctness plus the batch's mean predictive entropy."""
    logits = logits.detach()
    return (logits.argmax(dim=1) == targets), float(predictiveEntropy(logits).mean())


def runCell(batches, predict):
    """Adapt and classify a whole cell."""
    correct = 0
    total = 0
    entropies = []
    for batch, targets in batches:
        flags, entropy = scoreBatch(predict(batch, targets), targets)
        correct += int(flags.sum())
        total += len(batch)
        entropies.append(entropy)
    return {
        "images": total,
        "accuracy": correct / total,
        "meanEntropy": sum(entropies) / len(entropies),
    }


def sectionCells(corruptions, severities):
    """The (corruption, severity) product of two configs/splits.yaml lists."""
    return [(corruption, severity)
            for severity in severities
            for corruption in corruptions]


def cellList(split):
    """Cells for a one-section split, i.e. configs/splits.yaml validation."""
    return sectionCells(split["corruptions"], split["severities"])


def testCellList(test):
    """The 65 evaluation cells, in seen-then-unseen order.

    configs/splits.yaml test holds two sections because the seen corruptions are
    only scored at the severities the tuning cells never used, so the product of
    the whole section would add 10 cells that belong to tuning.
    """
    return (sectionCells(test["seenCorruption"]["corruptions"],
                         test["seenCorruption"]["severities"])
            + sectionCells(test["unseenCorruption"]["corruptions"],
                           test["unseenCorruption"]["severities"]))


def evaluateCell(root, cell, count, meanStd, device, batchSize, seed, predict):
    """One cell for one method, from reading the array to the accuracy."""
    corruption, severity = cell
    return runCell(cellBatches(root, corruption, severity, count, meanStd, device,
                                batchSize, seed), predict)


def evaluateCells(root, cells, count, meanStd, device, batchSize, seed, predict):
    """{cell -> result} for one method; the tuning runs only need the mean accuracy."""
    return {cell: evaluateCell(root, cell, count, meanStd, device, batchSize, seed, predict)
            for cell in cells}


def runStream(batches, predict, corruptionNames):
    """One pass over the stream with no reset, then per-corruption segment accuracy.

    worstSegment is the worst-case number the report quotes alongside the mean.
    """
    flags = []
    origins = []
    entropies = []
    for batch, targets, origin in batches:
        correct, entropy = scoreBatch(predict(batch, targets), targets)
        flags.append(correct)
        origins.append(origin)
        entropies.append(entropy)
    flags = torch.cat(flags)
    origins = torch.cat(origins)
    segments = {}
    for identifier, corruption in enumerate(corruptionNames):
        mask = origins == identifier
        segments[corruption] = {
            "accuracy": float(flags[mask].float().mean()),
            "images": int(mask.sum()),
        }
    return {
        "images": int(len(flags)),
        "accuracy": float(flags.float().mean()),
        "meanEntropy": sum(entropies) / len(entropies),
        "segments": segments,
        "worstSegment": min(segment["accuracy"] for segment in segments.values()),
    }
