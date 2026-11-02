"""Summary statistics for the tables."""

import numpy as np


def relativeErrorReduction(accuracy, sourceAccuracy):
    """Share of the source model's remaining error that adaptation removes."""
    return (accuracy - sourceAccuracy) / (1.0 - sourceAccuracy)


def meanStd(accuracies):
    """Mean and population standard deviation across seeds."""
    values = np.asarray(accuracies, dtype=float)
    return float(values.mean()), float(values.std())


def grouped(table, keyIndex):
    """{group -> mean accuracy} over the cells sharing one part of the cell key.

    keyIndex 0 groups by corruption, 1 by severity, so the per-corruption and
    per-severity views of a cell table come from the same pass.
    """
    values = {}
    for cell, result in table.items():
        values.setdefault(cell[keyIndex], []).append(result["accuracy"])
    return {key: float(np.mean(accuracy)) for key, accuracy in values.items()}


def restrict(samples, corruptions):
    """The seed tables limited to the given corruptions."""
    return {name: {seed: {cell: result for cell, result in table.items()
                          if cell[0] in corruptions}
                   for seed, table in perSeed.items()}
            for name, perSeed in samples.items()}


def cellMeans(perSeed):
    """[mean accuracy over the cells] for each seed table, in seed order."""
    return [float(np.mean([result["accuracy"] for result in table.values()]))
            for table in perSeed.values()]


def acrossSeeds(samples, keyIndex):
    """{method -> {group -> (mean, std)}}: group means per seed, then across seeds."""
    summary = {}
    for name, perSeed in samples.items():
        groups = [grouped(table, keyIndex) for table in perSeed.values()]
        summary[name] = {key: meanStd([group[key] for group in groups]) for key in groups[0]}
    return summary
