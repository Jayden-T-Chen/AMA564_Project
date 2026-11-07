"""Train the CIFAR-10 source model that every later experiment freezes.

The normalisation statistics are fit here on the training split only and stored
inside the checkpoint, so downstream runs cannot drift from what the backbone
was actually trained with.

Invoked as `run.runAll train` for the full 100 epochs, or `run.runAll train --epochs 2`
as a smoke test.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from configIO import loadConfig
from dataset.cifar10Source import trainingStatistics
from models.resNet18 import buildResNet18
from run.runUtils import resolveDevice, setSeed


def buildLoaders(paths, meanStd, seed, batchSize):
    mean, std = meanStd
    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return (
        DataLoader(
            CIFAR10(paths["cifar10Root"], train=True, download=False, transform=trainTransform),
            batch_size=batchSize, shuffle=True, num_workers=2,
            generator=torch.Generator().manual_seed(seed), persistent_workers=True,
        ),
        DataLoader(
            CIFAR10(paths["cifar10Root"], train=False, download=False, transform=testTransform),
            batch_size=256, shuffle=False, num_workers=2, persistent_workers=True,
        ),
    )


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        correct += int((model(images.to(device)).argmax(dim=1) == labels.to(device)).sum())
    return correct / len(loader.dataset)


def main(args=()):
    base = loadConfig("base")
    parser = argparse.ArgumentParser(prog="run.runAll train")
    parser.add_argument("--epochs", type=int, default=base["source"]["epochs"])
    epochs = parser.parse_args(list(args)).epochs
    source = base["source"]
    device = resolveDevice(base["deviceOrder"])
    setSeed(base["seed"])

    stats = trainingStatistics(base["paths"]["cifar10Root"])
    meanStd = (stats["mean"], stats["std"])
    print(f"fitted normalisation on train split: mean {[round(v,4) for v in stats['mean']]} "
          f"std {[round(v,4) for v in stats['std']]}")

    model = buildResNet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=source["lr"], momentum=source["momentum"],
        weight_decay=source["weightDecay"], nesterov=True,
    )
    trainLoader, testLoader = buildLoaders(base["paths"], meanStd, base["seed"], source["batchSize"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    curve = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        for images, labels in trainLoader:
            optimizer.zero_grad()
            functional.cross_entropy(model(images.to(device)), labels.to(device)).backward()
            optimizer.step()
        scheduler.step()
        testAccuracy = accuracy(model, testLoader, device)
        curve.append({
            "epoch": epoch,
            "testAccuracy": testAccuracy,
            "seconds": round(time.perf_counter() - start, 1),
        })
        print(f"epoch {epoch:3d}  test acc {testAccuracy * 100:5.2f}%  "
              f"lr {scheduler.get_last_lr()[0]:.5f}  {curve[-1]['seconds']}s", flush=True)

    checkpointDir = Path(base["paths"]["checkpointDir"])
    checkpointDir.mkdir(parents=True, exist_ok=True)
    checkpointPath = checkpointDir / f"resNet18Seed{base['seed']}.pt"
    torch.save({
        "stateDict": model.state_dict(),
        "seed": base["seed"],
        "normMean": stats["mean"],
        "normStd": stats["std"],
        "recipe": source,
        "epochsRun": epochs,
        "testAccuracy": curve[-1]["testAccuracy"],
    }, checkpointPath)

    resultsDir = Path(base["paths"]["resultsDir"])
    resultsDir.mkdir(parents=True, exist_ok=True)
    with open(resultsDir / "sourceTrain.json", "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": str(checkpointPath), "device": str(device),
                   "fittedNormalisation": stats, "curve": curve}, handle, indent=2)
    print(f"saved {checkpointPath} and results/sourceTrain.json")
