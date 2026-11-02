"""uint8 image blocks to normalised float tensors on the compute device."""

import numpy as np
import torch


def normalisationTensors(mean, std, device):
    shape = (1, 3, 1, 1)
    return (
        torch.tensor(mean, dtype=torch.float32, device=device).reshape(shape),
        torch.tensor(std, dtype=torch.float32, device=device).reshape(shape),
    )


def toDeviceTensor(images, device):
    """[n,32,32,3] uint8 -> [n,3,32,32] float32 (0-255 scale), on device.

    The result must stay contiguous: a train-mode BatchNorm backward on MPS
    aborts the process when its input has permuted strides.
    """
    array = np.array(images, dtype=np.uint8)
    tensor = torch.from_numpy(array).to(device)
    return tensor.permute(0, 3, 1, 2).float().contiguous()


def prepare(images, meanStd, device):
    """Normalise a uint8 block once per cell, on device."""
    mean, std = meanStd
    return toDeviceTensor(images, device).div_(255.0).sub_(mean).div_(std)
