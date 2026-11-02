"""TENT: entropy minimisation over BN affine parameters."""

import torch

from methods.losses import predictiveEntropy
from models.resNet18 import keepBnAffineOnly


def makeTentStep(model, learningRate, labelMask=False):
    """Return step(batch, targets) -> (mean entropy, updated parameter count).

    With labelMask only the samples the model already gets right *before* the
    update contribute to the entropy loss, and the mean is taken over those.
    That is the oracle upper bound on per-sample gating: it reads the labels, so
    it is a reference point and never a method.
    """
    model.train()  # BN normalises with the current batch statistics
    trainable, trainableCount = keepBnAffineOnly(model)
    optimizer = torch.optim.SGD(trainable, lr=learningRate)

    def step(batch, targets=None):
        optimizer.zero_grad()
        logits = model(batch)
        entropies = predictiveEntropy(logits)
        loss = entropies.mean()
        if labelMask:
            correct = (logits.argmax(dim=1) == targets).to(entropies.dtype)
            loss = (entropies * correct).sum() / correct.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
        return float(entropies.detach().mean()), trainableCount

    return step


def makeTentPredict(model, learningRate, labelMask=False):
    """predict(batch, targets) -> logits for the adapted model on that same batch.

    Reuses makeTentStep for the update, then reads the batch back out with the
    updated affine parameters. This is the posterior evaluation of the original
    TENT code: one entropy step per batch, accuracy measured after the step.
    """
    step = makeTentStep(model, learningRate, labelMask)

    def predict(batch, targets=None):
        step(batch, targets)
        with torch.no_grad():
            return model(batch)

    return predict
