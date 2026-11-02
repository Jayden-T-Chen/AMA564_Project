"""ResNet-18 for CIFAR-10 and the BN-affine-only interface every method shares."""

import torch
import torchvision.models as models

from torch.nn import BatchNorm2d


def buildResNet18(numClasses=10):
    # weights=None: trained from scratch on CIFAR-10, so no pretrained weights
    # or external data enter the study (disclosed in the report).
    return models.resnet18(weights=None, num_classes=numClasses)


def keepBnAffineOnly(model):
    """Freeze everything except BN scale and shift, and count what is left."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, BatchNorm2d):
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return trainable, sum(parameter.numel() for parameter in trainable)


def loadSourceModel(checkpointPath, device):
    """Rebuild the frozen backbone from a run/trainSource.py checkpoint.

    Returns (model, checkpoint); the checkpoint carries the normalisation
    statistics the backbone was actually trained with.
    """
    checkpoint = torch.load(checkpointPath, map_location=device)
    model = buildResNet18()
    model.load_state_dict(checkpoint["stateDict"])
    return model.to(device), checkpoint
