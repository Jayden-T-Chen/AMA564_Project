"""Online mixed-drift stream, and the batch-size sweep that runs over it.

Run:  .venv/bin/python -m run.runStream          # adaptation.batchSize only
      .venv/bin/python -m run.runStream sweep    # every adaptation.batchSweep value

Writes results/stream.csv, one row per method x batch x seed for the whole stream
plus one per corruption segment, and prints the mean accuracy and the worst
segment, which is the worst-case number the report quotes for the primary
setting. Each pass starts from the frozen checkpoint and a reseeded order, so the
methods see the same images in the same order; the padded batch order means a few
images are scored twice, exactly as in the static cells.

The stream is built only from unseen corruptions at severity 5
(configs/splits.yaml onlineStream), so nothing in it was used for tuning.
"""

import sys
import time

from configIO import loadConfig
from evaluation.metrics import meanStd
from evaluation.protocols import runStream, streamBatches
from models.resNet18 import loadSourceModel
from run.runMain import methodFactories
from run.runUtils import appendRows, setSeed
from run.trialDriver import buildContext

FIELD_NAMES = ["method", "batch", "segment", "seed", "images", "accuracy",
               "meanEntropy", "seconds"]


def streamRows(name, batch, seed, result, seconds):
    """One row for the whole stream plus one per corruption segment."""
    row = {"method": name, "batch": batch, "seed": seed, "meanEntropy": "", "seconds": seconds}
    return [dict(row, segment="all", images=result["images"],
                 accuracy=round(result["accuracy"], 6),
                 meanEntropy=round(result["meanEntropy"], 5))] + [
        dict(row, segment=corruption, images=segment["images"],
             accuracy=round(segment["accuracy"], 6))
        for corruption, segment in result["segments"].items()
    ]


def main():
    base, splits, gate = loadConfig("base"), loadConfig("splits"), loadConfig("gate")
    adaptation, stream = base["adaptation"], splits["onlineStream"]
    batches = adaptation["batchSweep"] if len(sys.argv) > 1 else [stream["batchSize"]]
    context = buildContext([], stream["imagesPerCorruption"], "stream.csv")
    context["csvPath"].unlink(missing_ok=True)
    factories = methodFactories(adaptation["lr"], gate, adaptation["tunedLr"])
    seeds = adaptation["seeds"]
    print(f"stream of {len(stream['corruptions'])} unseen corruptions x "
          f"{stream['imagesPerCorruption']} images at severity {stream['severity']}, "
          f"no reset between corruptions, batch sizes {batches}, {len(seeds)} seeds, "
          f"checkpoint {context['checkpointPath'].name}")

    results = {batch: {name: {} for name in factories} for batch in batches}
    for batch in batches:
        for seed in seeds:
            context["seed"] = seed
            for name, makePredict in factories.items():
                setSeed(seed)
                model, _ = loadSourceModel(context["checkpointPath"], context["device"])
                start = time.perf_counter()
                result = runStream(streamBatches(context["root"], stream, context["meanStd"],
                                                 context["device"], seed, batch),
                                   makePredict(model), stream["corruptions"])
                seconds = round(time.perf_counter() - start, 1)
                results[batch][name][seed] = result
                appendRows(context["csvPath"], streamRows(name, batch, seed, result, seconds),
                           FIELD_NAMES)
                print(f"batch {batch:3d}  seed {seed}  {name:16s} "
                      f"acc {result['accuracy'] * 100:6.2f}%  "
                      f"worst segment {result['worstSegment'] * 100:6.2f}%  {seconds}s",
                      flush=True)

    for batch in batches:
        print(f"\n== stream at batch {batch}, mean over {len(seeds)} seeds ==")
        print(f"{'method':16s}{'stream acc':>12s}{'std':>7s}{'worst segment':>15s}")
        for name, perSeed in results[batch].items():
            mean, std = meanStd([r["accuracy"] for r in perSeed.values()])
            worst, _ = meanStd([r["worstSegment"] for r in perSeed.values()])
            print(f"{name:16s}{mean * 100:11.2f}%{std * 100:6.2f}{worst * 100:14.2f}%")
        print(f"{'':16s}" + "".join(f"{corruption:>17s}" for corruption in stream["corruptions"]))
        for name, perSeed in results[batch].items():
            print(f"{name:16s}" + "".join(
                f"{meanStd([r['segments'][corruption]['accuracy'] for r in perSeed.values()])[0] * 100:16.2f}%"
                for corruption in stream["corruptions"]))
    print(f"\nrows in {context['csvPath']}")


if __name__ == "__main__":
    main()
