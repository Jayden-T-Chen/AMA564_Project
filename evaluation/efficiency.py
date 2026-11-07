"""Per-batch latency and memory on the run device."""

import math
import sys
import time

import torch


def allocatedMb(device):
    """Peak allocated memory in MB, or NaN where the platform exposes no reading.

    `resource` is imported here rather than at module level: Windows has no such
    module, and a top-level import would take down every command that reaches this
    file, including the report that only reads CSVs.
    """
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 2**20
    try:
        import resource
    except ImportError:
        return math.nan
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes here, Linux reports kilobytes. On CPU this is the whole
    # process peak, so it includes the interpreter and torch; mps and cuda report
    # tensors only. The three devices' numbers are not comparable to each other.
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


def measure(step, batches, maxBatches=50, warmBatches=5):
    """Run `step` over up to `maxBatches` batches, return timing and memory.

    `step` must return a python float (or a tensor whose value is read out),
    which already forces the device to finish its queue, so no extra sync.
    `peakAllocatedMb` stays NaN when no batch produced a reading.
    """
    peakMb = math.nan
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
        reading = allocatedMb(batch.device)
        if not math.isnan(reading):
            peakMb = reading if math.isnan(peakMb) else max(peakMb, reading)
        if index + 1 >= maxBatches:
            break
    return {
        "batches": counted,
        "latencyMsPerBatch": 1000 * elapsedSeconds / max(counted, 1),
        "imagesPerSecond": images / elapsedSeconds if elapsedSeconds else float("inf"),
        "peakAllocatedMb": peakMb,
    }
