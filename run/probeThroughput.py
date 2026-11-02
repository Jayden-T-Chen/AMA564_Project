"""Measure what this machine can do before committing to an experiment grid.

Run:  .venv/bin/python -m run.probeThroughput
Weights are random here; throughput does not depend on trained values.
"""

import torch

from configIO import loadConfig
from dataset.cifar10CLoader import cell
from dataset.preprocess import normalisationTensors, prepare
from evaluation.efficiency import measure
from methods.gatedTent import makeGatedTentStep
from methods.tent import makeTentStep
from models.resNet18 import buildResNet18, keepBnAffineOnly
from run.runUtils import batchOrder, resolveDevice, setSeed


def main():
    base = loadConfig("base")
    device = resolveDevice(base["deviceOrder"])
    setSeed(base["seed"])
    source = base["source"]
    batchSize = base["adaptation"]["batchSize"]

    model = buildResNet18().to(device)
    _, trainableCount = keepBnAffineOnly(model)
    meanStd = normalisationTensors(source["normMean"], source["normStd"], device)
    images, _ = cell(base["paths"]["cifar10cRoot"], "gaussian_noise", 5, 2000)
    data = prepare(images, meanStd, device)
    order = batchOrder(len(data), batchSize, base["seed"])
    batches = (data[order[i:i + batchSize]] for i in range(0, len(order), batchSize))

    with torch.no_grad():
        forwardOnly = measure(lambda batch: float(model(batch).sum().item()), batches)

    step = makeTentStep(model, base["adaptation"]["lr"])
    batches = (data[order[i:i + batchSize]] for i in range(0, len(order), batchSize))
    tent = measure(step, batches)
    # makeTentStep and makeGatedTentStep both put the model in train mode, which is
    # the configuration every run in this repo uses, so both numbers below are
    # measured with BN reading the current batch statistics.
    gatedStep = makeGatedTentStep(model, base["adaptation"]["lr"], loadConfig("gate"))
    batches = (data[order[i:i + batchSize]] for i in range(0, len(order), batchSize))
    gated = measure(gatedStep, batches)

    print(f"device={device} batchSize={batchSize} updatedParams={trainableCount}")
    print(f"forward only : {forwardOnly['imagesPerSecond']:7.1f} img/s  "
          f"{forwardOnly['latencyMsPerBatch']:6.2f} ms/batch  peak {forwardOnly['peakAllocatedMb']:7.1f} MB")
    print(f"TENT step    : {tent['imagesPerSecond']:7.1f} img/s  "
          f"{tent['latencyMsPerBatch']:6.2f} ms/batch  peak {tent['peakAllocatedMb']:7.1f} MB")
    print(f"consistency  : {gated['imagesPerSecond']:7.1f} img/s  "
          f"{gated['latencyMsPerBatch']:6.2f} ms/batch  peak {gated['peakAllocatedMb']:7.1f} MB")
    for name, count in (("65 test cells", 650000), ("severity-5 grid", 150000), ("online stream", 25000)):
        print(f"  {name:16s} = {count:7d} images -> "
              f"forward-only {count / forwardOnly['imagesPerSecond'] / 60:5.1f} min, "
              f"TENT {count / tent['imagesPerSecond'] / 60:5.1f} min, "
              f"consistency {count / gated['imagesPerSecond'] / 60:5.1f} min")


if __name__ == "__main__":
    main()
