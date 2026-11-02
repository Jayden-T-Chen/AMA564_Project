"""The one experiment pass shared by tuning, ablation and the main table."""

import time
from pathlib import Path

from configIO import loadConfig
from dataset.preprocess import normalisationTensors
from evaluation.protocols import evaluateCells
from models.resNet18 import loadSourceModel
from run.runUtils import appendRows, resolveDevice, setSeed


def buildContext(cells, count, csvName):
    """Device, frozen checkpoint, normalisation and output path for a set of cells."""
    base = loadConfig("base")
    adaptation = base["adaptation"]
    device = resolveDevice(base["deviceOrder"])
    seed = base["seed"]
    checkpointPath = Path(base["paths"]["checkpointDir"]) / f"resNet18Seed{seed}.pt"
    _, checkpoint = loadSourceModel(checkpointPath, device)
    return {
        "base": base, "adaptation": adaptation, "cells": cells, "count": count,
        "checkpointPath": checkpointPath, "checkpoint": checkpoint,
        "device": device, "seed": seed, "root": base["paths"]["cifar10cRoot"],
        "batchSize": adaptation["batchSize"],
        "meanStd": normalisationTensors(checkpoint["normMean"], checkpoint["normStd"], device),
        "csvPath": Path(base["paths"]["resultsDir"]) / csvName,
    }


def runTrial(context, row, makePredict, fieldNames):
    """Evaluate one configuration over context["cells"] from a fresh checkpoint.

    Reseeding before the pass is what pairs the trials: the batch order and every
    augmentation draw then repeat across configurations. `row` holds the
    descriptive fields and is merged with the measured ones for the CSV.
    Returns {"table": per-cell result, "mean": mean accuracy, "seconds": total}.
    """
    setSeed(context["seed"])
    model, _ = loadSourceModel(context["checkpointPath"], context["device"])
    start = time.perf_counter()
    table = evaluateCells(context["root"], context["cells"], context["count"],
                          context["meanStd"], context["device"], context["batchSize"],
                          context["seed"], makePredict(model))
    seconds = round(time.perf_counter() - start, 1)
    mean = sum(result["accuracy"] for result in table.values()) / len(table)
    appendRows(context["csvPath"], [
        dict(row, seed=context["seed"], corruption=corruption, severity=severity,
             images=result["images"], accuracy=round(result["accuracy"], 6),
             meanEntropy=round(result["meanEntropy"], 5), seconds=seconds)
        for (corruption, severity), result in table.items()
    ], fieldNames)
    return {"table": table, "mean": mean, "seconds": seconds}
