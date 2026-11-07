"""Main evaluation: the 65 test cells, six methods, three seeds.

Invoked as `run.runAll main`, or `run.runAll main oracle` for just those methods.

Writes results/mainTable.csv, one row per method x seed x cell, and prints the
report tables: mean accuracy over all 65 cells, the per-corruption matrix, the
severity curve of the unseen corruptions, and the worst-case numbers. Rows are
appended after each method x seed pass, so an interrupted run still keeps the
part it finished. Naming methods runs a subset and keeps whatever is already in
the CSV, which is how a late baseline joins a table that is already committed;
the printed tables always describe the whole file.

The cells follow adaptation.resetPerCell, so under the episodic default a cell is
read on nothing but itself and the seen/unseen split measures the split rather
than the position a cell holds in a chain. Each row also records the distance that
cell moved the parameters, which is the axis the matched-displacement control uses.

The fourth method is read from the configs/gate.yaml defaults, which is the one
place the final objective is defined and the file the ablation already uses.
A checkpoint exists only for base.seed, so the extra seeds vary the batch order
and the augmentation draws, not the source model.
"""

from collections import defaultdict

from configIO import loadConfig
from evaluation.metrics import (acrossSeeds, cellMeans, grouped, meanStd,
                                relativeErrorReduction, restrict)
from evaluation.protocols import testCellList
from methods.gatedTent import makeGatedTentPredict
from methods.tent import makeTentPredict
from run.runPilot import methodFactories as baselineFactories
from run.runUtils import readRows
from run.trialDriver import buildContext, runTrial

FIELD_NAMES = ["method", "seed", "corruption", "severity", "images", "accuracy",
               "meanEntropy", "displacementNorm", "seconds"]
OURS = "consistencyTent"


def methodFactories(learningRate, gate, tunedLr=None):
    """method name -> callable(model) -> predict(batch, targets), in report order.

    `tunedLr` adds TENT at the step size selected on the validation cells: the
    guidance asks for comparable tuning budgets, and the consistency term turns
    out to buy roughly what a larger step buys. `oracle` is TENT with the entropy
    loss restricted to the samples it already classifies correctly. It reads
    labels, so it is a reference point rather than a method, and it is one greedy
    rule rather than a bound over every rule that could select samples.
    """
    factories = dict(baselineFactories(learningRate))
    if tunedLr:
        factories["tentTuned"] = lambda model: makeTentPredict(model, tunedLr)
    factories[OURS] = lambda model: makeGatedTentPredict(model, learningRate, gate)
    factories["oracle"] = lambda model: makeTentPredict(model, learningRate, labelMask=True)
    return factories


def samplesFromRows(rows):
    """{method -> {seed -> {cell -> {"accuracy", "displacementNorm"}}}}."""
    samples = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        samples[row["method"]][int(row["seed"])][row["corruption"], int(row["severity"])] = {
            "accuracy": float(row["accuracy"]),
            # Absent in a CSV written before the displacement column existed; the
            # two no-update reference points genuinely sit at zero.
            "displacementNorm": float(row.get("displacementNorm") or 0.0),
        }
    return {name: {seed: table for seed, table in perSeed.items()}
            for name, perSeed in samples.items()}


def perSeedWorst(perSeed, keyIndex):
    """Worst case for each seed: over cells, or over corruption segments."""
    if keyIndex is None:
        return [min(result["accuracy"] for result in table.values())
                for table in perSeed.values()]
    return [min(grouped(table, keyIndex).values()) for table in perSeed.values()]


def meanDisplacement(table):
    """Mean distance one cell moves the parameters, over that table's cells."""
    values = [result["displacementNorm"] for result in table.values()]
    return sum(values) / len(values)


def printMeanTable(samples):
    source = cellMeans(samples["sourceOnly"])
    print("\n== mean over the 65 test cells, and the worst case ==")
    print(f"{'method':16s}{'mean acc':>10s}{'std':>7s}{'rel err red':>13s}"
          f"{'worst cell':>12s}{'worst corrp':>13s}{'path/cell':>11s}")
    for name, perSeed in samples.items():
        mean, std = meanStd(cellMeans(perSeed))
        worstCell = meanStd(perSeedWorst(perSeed, None))[0]
        worstCorruption = meanStd(perSeedWorst(perSeed, 0))[0]
        path = sum(meanDisplacement(table) for table in perSeed.values()) / len(perSeed)
        print(f"{name:16s}{mean * 100:9.2f}%{std * 100:6.2f}"
              f"{relativeErrorReduction(mean, sum(source) / len(source)) * 100:12.1f}%"
              f"{worstCell * 100:11.2f}%{worstCorruption * 100:12.2f}%"
              f"{path:11.3f}")


def printMatrix(header, summary, keys):
    names = list(summary)
    print(f"\n== {header} ==")
    print(f"{'':16s}" + "".join(f"{name:>17s}" for name in names))
    for key in keys:
        print(f"{str(key):16s}" + "".join(
            f"{summary[name][key][0] * 100:12.2f} ±{summary[name][key][1] * 100:4.1f}"
            for name in names))


def main(args=()):
    base, splits, gate = loadConfig("base"), loadConfig("splits"), loadConfig("gate")
    adaptation, test = base["adaptation"], splits["test"]
    cells = testCellList(test)
    context = buildContext(cells, test["imagesPerCell"], "mainTable.csv")
    factories = methodFactories(adaptation["lr"], gate, adaptation["tunedLr"])
    requested = list(args) or list(factories)
    if requested == list(factories):
        context["csvPath"].unlink(missing_ok=True)
    seeds = adaptation["seeds"]
    print(f"main table on {len(cells)} test cells x {test['imagesPerCell']} images, "
          f"{len(seeds)} seeds, checkpoint {context['checkpointPath'].name}, "
          f"batch {context['batchSize']}, lr {adaptation['lr']} "
          f"(tentTuned at {adaptation['tunedLr']}), "
          f"{OURS} at lambda {gate['consistencyGate']['weight']}, "
          f"{'episodic' if context['resetPerCell'] else 'chained'} cells\n"
          f"methods: {', '.join(requested)}")

    for seed in seeds:
        context["seed"] = seed
        for name, makePredict in factories.items():
            if name not in requested:
                continue
            result = runTrial(context, {"method": name, "seed": seed},
                              makePredict, FIELD_NAMES)
            print(f"seed {seed}  {name:16s} mean acc {result['mean'] * 100:6.2f}%  "
                  f"{result['seconds']}s", flush=True)

    samples = samplesFromRows(readRows(context["csvPath"]))
    printMeanTable(samples)
    printMatrix("accuracy per corruption, mean over its test cells",
                acrossSeeds(samples, 0), sorted({corruption for corruption, _ in cells}))
    printMatrix("accuracy per severity, unseen corruptions only",
                acrossSeeds(restrict(samples, set(test["unseenCorruption"]["corruptions"])), 1),
                test["unseenCorruption"]["severities"])
    print(f"\nrows in {context['csvPath']}")
