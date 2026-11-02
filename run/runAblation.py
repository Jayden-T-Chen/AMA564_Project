"""Variant sweeps: which part of the gain comes from which mechanism.

Run:  .venv/bin/python -m run.runAblation            # the gate ablation
      .venv/bin/python -m run.runAblation probe      # the step-size control

The block argument names a section of configs/gate.yaml holding a `cells` list
and a `variants` list, so the comparison cannot drift away from the file that
defines it. Output goes to results/<block>.csv, one row per variant x cell, and
the summary prints each variant against the block's `reference` row.

A variant may carry its own `lr` or `batch`; without one it uses the value from
configs/base.yaml adaptation. That is what lets the probe blocks change only the
step size or only the batch size while everything else stays on the frozen
defaults.

Every variant, TENT included, goes through the gated implementation: with both
`use` switches off the weights are all ones and the objective is mean predictive
entropy, which is TENT. Running the reference through the same code path is the
cheapest way to keep the difference attributable to the gates alone.
"""

import copy
import sys

from configIO import loadConfig, setPath
from evaluation.protocols import cellList
from methods.gatedTent import makeGatedTentPredict
from run.trialDriver import buildContext, runTrial

FIELD_NAMES = ["variant", "overrides", "lr", "batch", "seed", "corruption", "severity",
               "images", "accuracy", "meanEntropy", "seconds"]


def variantGate(gate, overrides):
    """A copy of the frozen gate config with this variant's keys replaced."""
    for path, value in overrides.items():
        setPath(gate, path, value)
    return gate


def main():
    block = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    splits, gate = loadConfig("splits"), loadConfig("gate")
    spec = gate[block]
    target = splits[spec["cells"]]
    context = buildContext(cellList(target), target["imagesPerCell"], f"{block}.csv")
    context["csvPath"].unlink(missing_ok=True)
    defaultLr = context["adaptation"]["lr"]
    defaultBatch = context["batchSize"]
    print(f"{block} on {len(context['cells'])} {spec['cells']} cells x {context['count']} "
          f"images, checkpoint {context['checkpointPath'].name}, seed {context['seed']}, "
          f"default batch {defaultBatch}, default lr {defaultLr}")

    results = {}
    settings = {}
    for variant in spec["variants"]:
        overrides = variant.get("overrides") or {}
        learningRate = variant.get("lr", defaultLr)
        context["batchSize"] = variant.get("batch", defaultBatch)
        name = variant["name"]
        label = ";".join(f"{path}={value}" for path, value in overrides.items()) or "defaults"
        trialGate = variantGate(copy.deepcopy(gate), overrides)
        result = runTrial(context, {"variant": name, "overrides": label, "lr": learningRate,
                                    "batch": context["batchSize"]},
                          lambda model, trialGate=trialGate, learningRate=learningRate:
                          makeGatedTentPredict(model, learningRate, trialGate), FIELD_NAMES)
        results[name] = result
        settings[name] = (learningRate, context["batchSize"])
        print(f"{name:22s} lr {learningRate:<7g} batch {context['batchSize']:<4d} "
              f"{label:52s} mean acc {result['mean'] * 100:6.2f}%", flush=True)

    referenceName = spec.get("reference", "tent")
    reference = results[referenceName]["table"]
    print(f"\n== {block} summary against the {referenceName} row ==")
    print(f"{'variant':22s}{'lr':>8s}{'batch':>7s}{'mean acc':>11s}{'delta':>8s}"
          f"{'cells up':>10s}{'cells down':>12s}")
    for name, result in results.items():
        deltas = [result["table"][cell]["accuracy"] - reference[cell]["accuracy"]
                  for cell in reference]
        meanDelta = 100 * sum(deltas) / len(deltas)
        learningRate, batchSize = settings[name]
        print(f"{name:22s}{learningRate:>8g}{batchSize:>7d}{result['mean'] * 100:10.2f}%"
              f"{meanDelta:+8.2f}{sum(1 for delta in deltas if delta > 0):>10d}"
              f"{sum(1 for delta in deltas if delta < 0):>12d}")
    print(f"\nrows in {context['csvPath']}")


if __name__ == "__main__":
    main()
