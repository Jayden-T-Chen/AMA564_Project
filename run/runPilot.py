"""Pilot gate: does the frozen TENT beat source-only on the validation cells?

Run:  .venv/bin/python -m run.runPilot

Writes results/pilot.csv, one row per method x cell, and prints the verdict
against configs/pilot.yaml. The validation cells are tuning data only; the
65 test cells are never read here.
"""

import time
from pathlib import Path

from configIO import loadConfig
from dataset.preprocess import normalisationTensors
from evaluation.metrics import relativeErrorReduction
from evaluation.protocols import cellList, evaluateCell
from methods.baselines import makeBnAdapt, makeSourceOnly
from methods.tent import makeTentPredict
from models.resNet18 import loadSourceModel
from run.runUtils import appendRows, resolveDevice, setSeed

FIELD_NAMES = ["method", "seed", "corruption", "severity",
               "images", "accuracy", "meanEntropy", "seconds"]


def methodFactories(learningRate):
    """method name -> callable(model) -> predict(batch, targets), in report order."""
    return {
        "sourceOnly": makeSourceOnly,
        "bnAdapt": makeBnAdapt,
        "tent": lambda model: makeTentPredict(model, learningRate),
    }


def printVerdict(accuracies, corruptions, cells, gate):
    means = {name: sum(table.values()) / len(table) for name, table in accuracies.items()}
    sourceMean = means["sourceOnly"]
    print("\n== mean over the validation cells ==")
    for name, mean in means.items():
        gain = (mean - sourceMean) * 100
        reduction = relativeErrorReduction(mean, sourceMean)
        print(f"{name:12s} acc {mean * 100:6.2f}%   +{gain:5.2f} pp   "
              f"relative error reduction {reduction * 100:5.1f}%")

    severities = sorted({severity for _, severity in cells})

    def cellMean(table, corruption):
        values = [table[corruption, severity] for severity in severities]
        return sum(values) / len(values)

    print("\n== per corruption, averaged over severities ==")
    print(f"{'corruption':20s}{'source':>9s}{'bn-adapt':>10s}{'tent':>9s}"
          f"{'tent-src':>11s}{'tent-bn':>10s}")
    perCorruption = {}
    for corruption in corruptions:
        row = {name: cellMean(accuracies[name], corruption)
               for name in ("sourceOnly", "bnAdapt", "tent")}
        perCorruption[corruption] = 100 * (row["tent"] - row["sourceOnly"])
        print(f"{corruption:20s}{row['sourceOnly'] * 100:8.2f}%{row['bnAdapt'] * 100:9.2f}%"
              f"{row['tent'] * 100:8.2f}%"
              f"{100 * (row['tent'] - row['sourceOnly']):+11.2f}"
              f"{100 * (row['tent'] - row['bnAdapt']):+9.2f} pp")
    print("tent-src is what the gate rule reads; tent-bn separates the entropy objective "
          "from batch-statistics recalibration.")

    meanGain = (means["tent"] - sourceMean) * 100
    worst = min(perCorruption.values())
    passed = meanGain >= gate["minMeanGainPp"] and worst >= -gate["maxPerCorruptionDropPp"]
    print(f"\n== pilot gate: mean gain {meanGain:+.2f} pp "
          f"(need >= {gate['minMeanGainPp']}), worst corruption {worst:+.2f} pp "
          f"(must not fall below -{gate['maxPerCorruptionDropPp']}) ==")
    print("PASS -> keep the gated-TENT plan" if passed
          else "FAIL -> switch to plan B (batch-size / mixed-stream empirical study)")


def main():
    base, splits, gate = loadConfig("base"), loadConfig("splits"), loadConfig("pilot")
    adaptation, validation = base["adaptation"], splits["validation"]
    device = resolveDevice(base["deviceOrder"])
    seed = base["seed"]
    checkpointPath = Path(base["paths"]["checkpointDir"]) / f"resNet18Seed{seed}.pt"

    # Read once just to take the fitted normalisation statistics with it.
    _, checkpoint = loadSourceModel(checkpointPath, device)
    meanStd = normalisationTensors(checkpoint["normMean"], checkpoint["normStd"], device)
    print(f"checkpoint {checkpointPath.name}: clean test accuracy "
          f"{checkpoint['testAccuracy'] * 100:.2f}% over {checkpoint['epochsRun']} epochs, "
          f"device {device}, seed {seed}, batch {adaptation['batchSize']}")

    cells = cellList(validation)
    csvPath = Path(base["paths"]["resultsDir"]) / "pilot.csv"
    csvPath.unlink(missing_ok=True)
    accuracies = {}
    for name, makePredict in methodFactories(adaptation["lr"]).items():
        setSeed(seed)
        model, _ = loadSourceModel(checkpointPath, device)
        predict = makePredict(model)
        accuracies[name] = {}
        for corruption, severity in cells:
            start = time.perf_counter()
            result = evaluateCell(base["paths"]["cifar10cRoot"], (corruption, severity),
                                  validation["imagesPerCell"], meanStd, device,
                                  adaptation["batchSize"], seed, predict)
            accuracies[name][corruption, severity] = result["accuracy"]
            appendRows(csvPath, [{
                "method": name, "seed": seed, "corruption": corruption, "severity": severity,
                "images": result["images"], "accuracy": round(result["accuracy"], 6),
                "meanEntropy": round(result["meanEntropy"], 5),
                "seconds": round(time.perf_counter() - start, 1),
            }], FIELD_NAMES)
            print(f"{name:12s} {corruption:18s} sev{severity}  "
                  f"acc {result['accuracy'] * 100:6.2f}%  H {result['meanEntropy']:.4f}  "
                  f"{result['images']} imgs", flush=True)

    printVerdict(accuracies, validation["corruptions"], cells, gate)
    print(f"\nrows written to {csvPath}")


if __name__ == "__main__":
    main()
