"""Loss terms shared by TENT and gated-TENT."""

import torch
import torch.nn.functional as functional


def predictiveEntropy(logits):
    """Per-sample predictive entropy in nats, shape [n]."""
    logProbabilities = functional.log_softmax(logits, dim=1)
    return -(logProbabilities.exp() * logProbabilities).sum(dim=1)


def symmetricKl(logitsA, logitsB):
    """Per-sample disagreement between two views in nats, shape [n]."""
    logA = functional.log_softmax(logitsA, dim=1)
    logB = functional.log_softmax(logitsB, dim=1)
    forward = (logA.exp() * (logA - logB)).sum(dim=1)
    reverse = (logB.exp() * (logB - logA)).sum(dim=1)
    return 0.5 * (forward + reverse)
