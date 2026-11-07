"""Variant sweeps: which part of the gain comes from which mechanism.

Invoked as `run.runAll gate <block>`. The block argument names a section of
configs/gate.yaml holding a `cells` list and a `variants` list, so the comparison cannot
drift away from the file that defines it. Output goes to results/<block>.csv, one row per
variant x cell x seed.

A variant may carry its own `lr`, `batch` or `seeds`; without one it uses the block's
value, and a block without one uses configs/base.yaml. That is what lets the ladder change
only the step size, or run its batch-16 rows at one seed while the batch-64 rows that
carry the report's conclusion run at three.

Every variant, TENT included, goes through the gated implementation: with both
`use` switches off the weights are all ones and the objective is mean predictive
entropy, which is TENT. Running the reference through the same code path is the
cheapest way to keep the difference attributable to the gates alone.

Each variant's rows are its own, so the CSV keeps one row per seed and the figures can
spread a comparison over them. The summary prints each variant's accuracy averaged over
its seeds, against that batch size's own reference row, together with the distance its
parameters travel over one cell -- a variant that buys accuracy by travelling further is a
step-size effect, and the two numbers are what tell those apart.
"""

import copy
from collections import defaultdict

from configIO import loadConfig, setPath
from evaluation.protocols import cellList
from methods.gatedTent import makeGatedTentPredict
from run.runMain import meanDisplacement
from run.trialDriver import buildContext, runTrial

FIELD_NAMES = ["variant", "overrides", "lr", "batch", "seed", "corruption", "severity",
               "images", "accuracy", "meanEntropy", "displacementNorm", "seconds"]


def variantGate(gate, overrides):
    """A copy of the frozen gate config with this variant's keys replaced."""
    for path, value in overrides.items():
        setPath(gate, path, value)
    return gate


def averagedTable(tables):
    """One row per cell, averaging the seeds that produced it.

    The summary and the figures read a variant as one curve, so the seed averaging
    happens here rather than in each of them. Displacement averages alongside
    accuracy because the matched-displacement control compares the two.
    """
    return {cell: {"accuracy": sum(table[cell]["accuracy"] for table in tables) / len(tables),
                   "displacementNorm": sum(table[cell]["displacementNorm"]
                                           for table in tables) / len(tables)}
            for cell in tables[0]}


def main(args=()):
    block = args[0] if args else "ablation"
    splits, gate = loadConfig("splits"), loadConfig("gate")
    spec = gate[block]
    target = splits[spec["cells"]]
    context = buildContext(cellList(target), target["imagesPerCell"], f"{block}.csv")
    context["csvPath"].unlink(missing_ok=True)
    defaultLr = context["adaptation"]["lr"]
    defaultBatch = context["batchSize"]
    blockSeeds = spec.get("seeds", [context["seed"]])
    print(f"{block} on {len(context['cells'])} {spec['cells']} cells x {context['count']} "
          f"images, checkpoint {context['checkpointPath'].name}, seeds {blockSeeds}, "
          f"default batch {defaultBatch}, default lr {defaultLr}, "
          f"{'episodic' if context['resetPerCell'] else 'chained'} cells")

    results = {}
    settings = {}
    for variant in spec["variants"]:
        overrides = variant.get("overrides") or {}
        learningRate = variant.get("lr", defaultLr)
        context["batchSize"] = variant.get("batch", defaultBatch)
        seeds = variant.get("seeds", blockSeeds)
        name = variant["name"]
        label = ";".join(f"{path}={value}" for path, value in overrides.items()) or "defaults"
        trialGate = variantGate(copy.deepcopy(gate), overrides)
        tables = []
        seconds = 0.0
        for seed in seeds:
            context["seed"] = seed
            result = runTrial(context, {"variant": name, "overrides": label,
                                        "lr": learningRate, "batch": context["batchSize"]},
                              lambda model, trialGate=trialGate, learningRate=learningRate:
                              makeGatedTentPredict(model, learningRate, trialGate), FIELD_NAMES)
            tables.append(result["table"])
            seconds += result["seconds"]
            if len(seeds) > 1:
                print(f"  {name} seed {seed}: mean acc {result['mean'] * 100:6.2f}%", flush=True)
        table = averagedTable(tables)
        results[name] = {"table": table,
                         "mean": sum(row["accuracy"] for row in table.values()) / len(table),
                         "seconds": round(seconds, 1)}
        settings[name] = (learningRate, context["batchSize"])
        print(f"{name:22s} lr {learningRate:<7g} batch {context['batchSize']:<4d} "
              f"{len(seeds)} seed{'s' if len(seeds) > 1 else ''}        "
              f"{label:52s} mean acc {results[name]['mean'] * 100:6.2f}%  "
              f"path/cell {meanDisplacement(table):6.3f}", flush=True)

    referenceName = spec.get("reference", "tent")
    print(f"\n== {block} summary, each batch size against its own reference row ==")
    print(f"{'variant':26s}{'lr':>8s}{'batch':>7s}{'mean acc':>11s}{'delta':>8s}"
          f"{'path/cell':>11s}{'cells up':>10s}{'cells down':>12s}")
    bySize = defaultdict(list)
    for name, result in results.items():
        bySize[settings[name][1]].append(name)
    for size in sorted(bySize):
        # A block that spans batch sizes needs one reference per size: batch 16 holds
        # four times the update steps of batch 64, so a delta across the two measures
        # the batch and the variant at once. `lr1` is that name at every size the
        # ladder reaches; blocks without that convention fall back to the declared one.
        ownReference = next((name for name in bySize[size] if name.endswith("_lr1")), None)
        reference = results[ownReference or referenceName]["table"]
        for name in sorted(bySize[size], key=lambda name: results[name]["mean"]):
            deltas = [results[name]["table"][cell]["accuracy"] - reference[cell]["accuracy"]
                      for cell in reference]
            meanDelta = 100 * sum(deltas) / len(deltas)
            learningRate, batchSize = settings[name]
            print(f"{name:26s}{learningRate:>8g}{batchSize:>7d}{results[name]['mean'] * 100:10.2f}%"
                  f"{meanDelta:+8.2f}{meanDisplacement(results[name]['table']):11.3f}"
                  f"{sum(1 for delta in deltas if delta > 0):>10d}"
                  f"{sum(1 for delta in deltas if delta < 0):>12d}")
        if ownReference:
            print(f"{'':26s}{'':8s}{'':7s}{'(delta is against ' + ownReference + ')':>30s}")
    print(f"\nrows in {context['csvPath']}")
