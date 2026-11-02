"""Sequential gate tuning on the validation cells: fraction first, then lambda.

Run:  .venv/bin/python -m run.runTuning

Writes results/tuning.csv, one row per trial x cell, and prints the mean
accuracy per trial over the 10 validation cells. The stages and candidate
values come from configs/gate.yaml tuning.stages, so this run is a lookup, not
a search: 3 values of keepFraction with lambda frozen, then 3 values of lambda
at the best keepFraction. One TENT pass is included as the number the gate has
to beat. The test cells are never read here.

Whatever wins is written back into configs/gate.yaml, because that file is what
every later run reads.
"""

import copy

from configIO import getPath, loadConfig, setPath
from evaluation.protocols import cellList
from methods.gatedTent import makeGatedTentPredict
from methods.tent import makeTentPredict
from run.trialDriver import buildContext, runTrial

FIELD_NAMES = ["trial", "parameter", "value", "seed", "corruption", "severity",
               "images", "accuracy", "meanEntropy", "seconds"]


def main():
    splits, gate = loadConfig("splits"), loadConfig("gate")
    validation = splits["validation"]
    context = buildContext(cellList(validation), validation["imagesPerCell"], "tuning.csv")
    context["csvPath"].unlink(missing_ok=True)
    learningRate = context["adaptation"]["lr"]
    checkpoint = context["checkpoint"]
    print(f"tuning on {len(context['cells'])} validation cells x {context['count']} images, "
          f"checkpoint {context['checkpointPath'].name} "
          f"(clean test {checkpoint['testAccuracy'] * 100:.2f}%), "
          f"seed {context['seed']}, batch {context['batchSize']}")

    def trial(label, parameter, value, gated=None):
        makePredict = (lambda model: makeTentPredict(model, learningRate)) if gated is None else \
            (lambda model, gated=gated: makeGatedTentPredict(model, learningRate, gated))
        mean = runTrial(context, {"trial": label, "parameter": parameter, "value": value},
                        makePredict, FIELD_NAMES)["mean"]
        print(f"{label:34s} mean acc {mean * 100:6.2f}%", flush=True)
        return mean

    means = {"tent": trial("tent", "-", "-")}
    for stage in gate["tuning"]["stages"]:
        override = stage["override"]
        bestValue, bestMean = None, -1.0
        for value in getPath(gate, stage["valuesFrom"]):
            trialGate = setPath(copy.deepcopy(gate), override, value)
            label = f"{override}={value}"
            mean = trial(label, override, value, trialGate)
            means[label] = mean
            if mean > bestMean:
                bestValue, bestMean = value, mean
        setPath(gate, override, bestValue)
        print(f"  -> best {override} = {bestValue} at mean accuracy {bestMean * 100:.2f}%")

    print("\n== tuning summary, mean accuracy over the validation cells ==")
    for label, mean in means.items():
        print(f"{label:34s} {mean * 100:6.2f}%   {100 * (mean - means['tent']):+5.2f} pp vs TENT")
    print(f"\nselected keepFraction = {getPath(gate, 'entropyGate.keepFraction')} and "
          f"weight = {getPath(gate, 'consistencyGate.weight')}; those belong in "
          f"configs/gate.yaml because every later run reads it from there. "
          f"results/tuning.csv keeps every cell.")


if __name__ == "__main__":
    main()
