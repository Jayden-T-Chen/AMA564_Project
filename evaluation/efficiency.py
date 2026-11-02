"""Per-batch latency and memory on the run device."""

import resource
import time

import torch


def allocatedMb(device):
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 2**20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure(step, batches, maxBatches=50, warmBatches=5):
    """Run `step` over up to `maxBatches` batches, return timing and memory.

    `step` must return a python float (or a tensor whose value is read out),
    which already forces the device to finish its queue, so no extra sync.
    """
    peakMb = 0.0
    elapsedSeconds = 0.0
    counted = 0
    images = 0
    for index, batch in enumerate(batches):
        start = time.perf_counter()
        step(batch)
        duration = time.perf_counter() - start
        if index >= warmBatches:
            elapsedSeconds += duration
            counted += 1
            images += len(batch)
        peakMb = max(peakMb, allocatedMb(batch.device))
        if index + 1 >= maxBatches:
            break
    return {
        "batches": counted,
        "latencyMsPerBatch": 1000 * elapsedSeconds / max(counted, 1),
        "imagesPerSecond": images / elapsedSeconds if elapsedSeconds else float("inf"),
        "peakAllocatedMb": peakMb,
    }
