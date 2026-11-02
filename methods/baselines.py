"""The two no-update reference points every adaptation method is compared to."""

import torch


def makeForwardOnly(model, training):
    """Freeze everything and return predict(batch, targets) -> logits, no gradient step.

    With training=True the BatchNorm layers normalise by the current batch
    statistics. Their running statistics still get updated as a side effect,
    which is harmless here because a train-mode forward never reads them.

    `targets` is the second argument every method's predict takes because the
    evaluation loop hands it out; a reference point never reads it.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if training:
        model.train()
    else:
        model.eval()

    def predict(batch, targets=None):
        with torch.no_grad():
            return model(batch)

    return predict


def makeSourceOnly(model):
    """Source-only: BN uses the running statistics from source training."""
    return makeForwardOnly(model, training=False)


def makeBnAdapt(model):
    """BN-adapt (AdaBN): batch statistics, weights left exactly as trained."""
    return makeForwardOnly(model, training=True)
