"""How the two terms of the objective compare in gradient size, at the source point.

Invoked as `run.runAll gradient <block>`, over any block named in configs/gate.yaml.

Writes results/gradientScale-<block>.csv, one row per variant, holding
||grad mean(H)||, ||grad lambda.mean(d)|| and ||grad L|| for the objective the
optimiser actually sees. The ratio of the first two is why lambda behaves like a
step-size multiplier: at the frozen checkpoint the consistency term already
contributes most of the gradient the update takes.

These are read at the source parameters and nothing is stepped, so they describe
the start of adaptation rather than its whole trajectory. The distance a run
actually travels is not estimated from them -- every accuracy CSV records
`displacementNorm`, ||theta_after - theta_before|| per cell, measured on the run
itself, and that measured column is what the displacement figure plots. This file
explains the mechanism; it does not carry the axis.

Cells are prepared one at a time and reused across the variants at the same batch
size, so the probe holds one cell block on the device rather than every cell of
every batch size. Reseeding before each variant replays the same batches and the
same augmentation draws for all of them.
"""

import copy
import math

import torch

from configIO import loadConfig
from dataset.cifar10CLoader import cell
from dataset.preprocess import prepare
from evaluation.protocols import cellList
from methods.gatedTent import gatedObjective
from models.resNet18 import keepBnAffineOnly, loadSourceModel
from run.runAblation import variantGate
from run.runUtils import appendRows, batchOrder, setSeed
from run.trialDriver import buildContext

FIELD_NAMES = ["block", "variant", "lr", "batchSize", "lambdaWeight", "batchesPerCell",
               "entropyGradNorm", "consistencyGradNorm", "objectiveGradNorm", "termRatio"]


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


def probeBatches(prepared, targets, order, limit, batchSize):
    """The first `limit` batches of one prepared cell, as (images, targets) pairs."""
    return [(prepared[order[start:start + batchSize]], targets[order[start:start + batchSize]])
            for start in range(0, min(len(order), limit * batchSize), batchSize)]


def main(args=()):
    blockName = args[0] if args else "ladder"
    splits, gate = loadConfig("splits"), loadConfig("gate")
    spec = gate[blockName]
    context = buildContext(cellList(splits[spec["cells"]]),
                           splits[spec["cells"]]["imagesPerCell"],
                           f"gradientScale-{blockName}.csv")
    context["csvPath"].unlink(missing_ok=True)
    adaptation = context["adaptation"]
    limit = adaptation["gradientProbeBatches"]
    defaultLr, defaultBatch = adaptation["lr"], context["batchSize"]

    settings = [(variant["name"],
                 variant.get("lr", defaultLr),
                 variant.get("batch", defaultBatch),
                 variantGate(copy.deepcopy(gate), variant.get("overrides") or {}))
                for variant in spec["variants"]]
    model, _ = loadSourceModel(context["checkpointPath"], context["device"])
    trainable, _ = keepBnAffineOnly(model)
    print(f"gradient scale for the {blockName} block: {limit} batches from each of "
          f"{len(context['cells'])} {spec['cells']} cells, checkpoint "
          f"{context['checkpointPath'].name}, the same batches for every variant")

    # One cell block on the device at a time, reused by every variant at that batch size.
    accumulations = {name: [] for name, _, _, _ in settings}
    for batchSize in sorted({size for _, _, size, _ in settings}):
        order = batchOrder(context["count"], batchSize, context["seed"])
        for corruption, severity in context["cells"]:
            images, labels = cell(context["root"], corruption, severity, context["count"])
            prepared = prepare(images, context["meanStd"], context["device"])
            targets = torch.as_tensor(labels.astype("int64"), device=context["device"])
            batches = probeBatches(prepared, targets, order, limit, batchSize)
            for name, _learningRate, size, trialGate in settings:
                if size != batchSize:
                    continue
                # Reseeding per variant replays the same augmentation draws for all of them.
                setSeed(context["seed"])
                accumulations[name] += [measureBatch(model, trainable, batch, trialGate)
                                        for batch, _ in batches]
            del prepared, targets, batches

    for name, learningRate, batchSize, trialGate in settings:
        terms = accumulations[name]
        mean = [sum(term[position] for term in terms) / len(terms) for position in range(3)]
        consistency = trialGate["consistencyGate"]
        row = {"block": blockName, "variant": name, "lr": learningRate, "batchSize": batchSize,
               "lambdaWeight": consistency["weight"] if consistency["use"] else 0.0,
               "batchesPerCell": math.ceil(len(batchOrder(
                   context["count"], batchSize, context["seed"])) / batchSize),
               "entropyGradNorm": round(mean[0], 6),
               "consistencyGradNorm": round(mean[1], 6),
               "objectiveGradNorm": round(mean[2], 6),
               "termRatio": round(mean[1] / mean[0], 4) if mean[0] else 0.0}
        appendRows(context["csvPath"], [row], FIELD_NAMES)
        print(f"{name:26s} lr {learningRate:<8g} batch {batchSize:<4d} "
              f"lambda {row['lambdaWeight']:<5g} ||gH|| {row['entropyGradNorm']:6.3f}  "
              f"||gd|| {row['consistencyGradNorm']:7.3f}  "
              f"||g obj|| {row['objectiveGradNorm']:7.3f}  "
              f"ratio {row['termRatio']:6.2f}", flush=True)

    print(f"\nrows in {context['csvPath']}")
