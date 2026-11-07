"""Gated TENT: the entropy update only listens to the samples we trust.

Two gates, both read from configs/gate.yaml:

  entropy gate      hard mask, keep the lowest-entropy `keepFraction` of a batch
  consistency gate  soft weight w_i = max(1 - normalised disagreement, floor)

The update is `sum_i w_i (H_i + lambda * d_i) / sum_i w_i` with lambda =
consistencyGate.weight. The weights are detached, so they select and scale the
gradient but never contribute to it. With keepFraction 1.0 and the consistency
gate switched off this reduces to mean predictive entropy, i.e. exactly TENT,
which is the equivalence check the ablation depends on.
"""

import torch
import torch.nn.functional as functional

from methods.losses import predictiveEntropy, symmetricKl
from models.resNet18 import keepBnAffineOnly


def makeView(images, cropScale):
    """One label-preserving view: per-image random crop back to 32x32, then flip.

    Both operations are geometric on purpose: colour and blur transforms would
    resemble the corruptions under test. The crop is a single batched
    grid_sample, one affine window per image, so a batch of 64 costs one kernel
    and the code stays identical on MPS, CUDA and CPU.
    """
    count, _, height, width = images.shape
    scale = torch.rand(count, device=images.device) * (cropScale[1] - cropScale[0]) + cropScale[0]
    # The window centre may move by at most half the margin it leaves behind.
    shift = (2.0 * torch.rand(2, count, device=images.device) - 1.0) * (1.0 - scale)
    theta = torch.zeros(count, 2, 3, device=images.device)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = shift[0]
    theta[:, 1, 2] = shift[1]
    grid = functional.affine_grid(theta, (count, 3, height, width), align_corners=False)
    cropped = functional.grid_sample(images, grid, align_corners=False)
    flipped = torch.rand(count, 1, 1, 1, device=images.device) < 0.5
    return torch.where(flipped, cropped.flip(-1), cropped)


def normaliseDisagreement(disagreement):
    """Scale a batch's disagreement to 0-1 so the floor is severity-independent."""
    low = disagreement.min()
    span = (disagreement.max() - low).clamp_min(1e-6)
    return (disagreement - low) / span


def sampleWeights(entropies, disagreement, gate):
    """Per-sample gradient weights, 1 when both gates are off."""
    weights = torch.ones_like(entropies)
    consistency = gate["consistencyGate"]
    if consistency["use"] and consistency["inWeights"]:
        reduced = 1.0 - normaliseDisagreement(disagreement)
        weights = weights * reduced.clamp_min(consistency["weightFloor"])
    if gate["entropyGate"]["use"]:
        cutoff = torch.quantile(entropies, gate["entropyGate"]["keepFraction"])
        weights = weights * (entropies <= cutoff)
    return weights.detach()


def gatedObjective(model, batch, gate):
    """The loss one update uses, with the per-sample entropies and disagreement.

    Split out of the step so `run.runAll gradient` can take gradients of the very
    expression the optimiser sees instead of keeping a second copy of it.
    """
    logits = model(batch)
    entropies = predictiveEntropy(logits)
    consistency = gate["consistencyGate"]
    terms = entropies
    disagreement = None
    if consistency["use"]:
        viewLogits = model(makeView(batch, consistency["cropScale"]))
        disagreement = symmetricKl(logits, viewLogits)
        if consistency["inLoss"]:
            terms = entropies + consistency["weight"] * disagreement
    weights = sampleWeights(entropies, disagreement, gate)
    denominator = weights.sum() if gate["loss"]["normaliseByWeights"] else len(weights)
    return (weights * terms).sum() / denominator, entropies, disagreement


def makeGatedTentStep(model, learningRate, gate):
    """step(batch) -> (loss value, updated parameter count), same contract as TENT.

    The consistency term needs a second forward on an augmented view, so this
    costs one extra pass per batch over TENT.
    """
    model.train()
    trainable, trainableCount = keepBnAffineOnly(model)
    optimizer = torch.optim.SGD(trainable, lr=learningRate)

    def step(batch):
        loss, _, _ = gatedObjective(model, batch, gate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach()), trainableCount

    return step


def makeGatedTentPredict(model, learningRate, gate):
    """predict(batch, targets) -> logits after one gated update, on the real batch.

    Same posterior convention as makeTentPredict, and it reuses the step above.
    The targets argument is accepted and dropped: what makes this method cheap is
    that nothing here reads a label, which is exactly what the oracle measures.
    """
    step = makeGatedTentStep(model, learningRate, gate)

    def predict(batch, targets=None):
        step(batch)
        with torch.no_grad():
            return model(batch)

    return predict
