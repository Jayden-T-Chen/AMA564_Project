"""How large a step each configuration actually takes, measured on the validation cells.

Run:  .venv/bin/python -m run.runGradientScale            # the stepControl block
      .venv/bin/python -m run.runGradientScale smallBatch  # any variant block

Writes results/gradientScale-<block>.csv, one row per variant. The accuracy ladders in
results/probe.csv, results/stepControl.csv and results/smallBatch.csv show the consistency
term tracking TENT's learning rate, which leaves the question of whether they are one
mechanism or two. The quantity that decides it is the size of the update: SGD moves the
affine parameters by lr * ||grad L|| per batch, and a cell accumulates one such move per
batch, so this measures both.

entropyGradNorm and consistencyGradNorm are the two terms on their own, at their own scale:
mean ||grad mean(H)|| and mean ||grad lambda.mean(d)||. objectiveGradNorm is the norm of
the gradient the optimiser actually sees, and cellPathLength is that times lr times the
number of batches in a cell: the distance the parameters travel over one cell.

The sample is the first adaptation.gradientProbeBatches batches of every cell in the block,
the same batches for every variant. Nothing is stepped; only BN's running statistics move as
a side effect of the forward passes, and no method in this repo reads running statistics in
train mode.
"""

import copy
import math
import sys
from itertools import islice

import torch

from configIO import loadConfig
from evaluation.protocols import cellBatches, cellList
from methods.gatedTent import gatedObjective
from models.resNet18 import keepBnAffineOnly, loadSourceModel
from run.runAblation import variantGate
from run.runUtils import appendRows, batchOrder, setSeed
from run.trialDriver import buildContext

FIELD_NAMES = ["block", "variant", "lr", "batchSize", "lambdaWeight", "batchesPerCell",
               "entropyGradNorm", "consistencyGradNorm", "objectiveGradNorm",
               "meanStepNorm", "cellPathLength"]


def gradNorm(loss, trainable):
    """||grad loss|| over the trainable parameters, graph kept for the next call."""
    gradients = torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum(g.detach().pow(2).sum() for g in gradients if g is not None)))


def measureBatch(model, trainable, batch, gate):
    """[entropy, consistency, objective] gradient norms for one batch."""
    model.train()
    consistency = gate["consistencyGate"]
    loss, entropies, disagreement = gatedObjective(model, batch, gate)
    return [gradNorm(entropies.mean(), trainable),
            gradNorm(consistency["weight"] * disagreement.mean(), trainable)
            if disagreement is not None else 0.0,
            gradNorm(loss, trainable)]


def cellSamples(context, batchSize, limit):
    """[(corruption, severity, sampled batches)] at one batch size."""
    return [(corruption, severity, list(islice(cellBatches(
        context["root"], corruption, severity, context["count"], context["meanStd"],
        context["device"], batchSize, context["seed"]), limit)))
        for corruption, severity in context["cells"]]


def main():
    block = sys.argv[1] if len(sys.argv) > 1 else "stepControl"
    splits, gate = loadConfig("splits"), loadConfig("gate")
    spec = gate[block]
    context = buildContext(cellList(splits[spec["cells"]]),
                           splits[spec["cells"]]["imagesPerCell"], f"gradientScale-{block}.csv")
    context["csvPath"].unlink(missing_ok=True)
    adaptation = context["adaptation"]
    limit = adaptation["gradientProbeBatches"]
    names = [variant["name"] for variant in spec["variants"]]
    settings = [(variant.get("lr", adaptation["lr"]),
                 variant.get("batch", context["batchSize"]),
                 variantGate(copy.deepcopy(gate), variant.get("overrides") or {}))
                for variant in spec["variants"]]
    model, _ = loadSourceModel(context["checkpointPath"], context["device"])
    trainable, _ = keepBnAffineOnly(model)
    print(f"gradient scale for the {block} block: {limit} batches from each of "
          f"{len(context['cells'])} {spec['cells']} cells, checkpoint "
          f"{context['checkpointPath'].name}, the same batches for every variant")

    samples = {}
    results = {}
    for name, (learningRate, batchSize, trialGate) in zip(names, settings):
        if batchSize not in samples:
            samples[batchSize] = cellSamples(context, batchSize, limit)
        terms = []
        for _, _, batches in samples[batchSize]:
            # Reseeding per cell replays the same augmentation draws for every variant.
            setSeed(context["seed"])
            for batch, _ in batches:
                terms.append(measureBatch(model, trainable, batch, trialGate))
        results[name] = [sum(term[position] for term in terms) / len(terms)
                         for position in range(3)]

    for name, (learningRate, batchSize, trialGate) in zip(names, settings):
        terms = results[name]
        batchesPerCell = math.ceil(len(batchOrder(context["count"], batchSize,
                                                 context["seed"])) / batchSize)
        consistency = trialGate["consistencyGate"]
        row = {"block": block, "variant": name, "lr": learningRate, "batchSize": batchSize,
               "lambdaWeight": consistency["weight"] if consistency["use"] else 0.0,
               "batchesPerCell": batchesPerCell,
               "entropyGradNorm": round(terms[0], 6),
               "consistencyGradNorm": round(terms[1], 6),
               "objectiveGradNorm": round(terms[2], 6),
               "meanStepNorm": round(learningRate * terms[2], 8),
               "cellPathLength": round(learningRate * terms[2] * batchesPerCell, 6)}
        appendRows(context["csvPath"], [row], FIELD_NAMES)
        print(f"{name:26s} lr {learningRate:<7g} batch {batchSize:<4d} "
              f"lambda {row['lambdaWeight']:<5g} ||gH|| {row['entropyGradNorm']:6.3f}  "
              f"||gd|| {row['consistencyGradNorm']:7.3f}  "
              f"||g obj|| {row['objectiveGradNorm']:7.3f}  step {row['meanStepNorm']:.4f}  "
              f"path/cell {row['cellPathLength']:.3f}", flush=True)

    print(f"\nrows in {context['csvPath']}")


if __name__ == "__main__":
    main()
