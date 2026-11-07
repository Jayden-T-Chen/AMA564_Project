"""Figures for the report, drawn from the CSVs under results/.

Invoked as `run.runAll figures`, which the default `run.runAll` also does.

Every figure reads the rows the report quotes, so a number in the text and the point in
the picture come from the same CSV. A figure whose input has not been run yet is skipped
with a note, which lets this run at any point in the pipeline.

Output: results/figures/<name>.png.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configIO import loadConfig

METHOD_ORDER = ["sourceOnly", "bnAdapt", "tent", "tentTuned", "consistencyTent", "oracle"]


def resultPath(name):
    return Path(loadConfig("base")["paths"]["resultsDir"]) / name


def readRows(name):
    """The rows of results/<name>, or None when that run has not happened yet."""
    path = resultPath(name)
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def writeFigure(figure, name, note=""):
    directory = resultPath("figures")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"wrote {path}" + (f"  [{note}]" if note else ""))


def coverage(rows):
    """What the CSV actually holds, so a figure drawn mid-run says so out loud."""
    methods = {row.get("method", row.get("variant")) for row in rows}
    return f"{len(rows)} rows, {len(methods)} methods x {len({row['seed'] for row in rows})} seeds"


def cellAccuracy(rows):
    """{variant -> mean accuracy over its cells}, averaging seeds first."""
    cells = defaultdict(lambda: defaultdict(list))
    key = "variant" if "variant" in rows[0] else "method"
    for row in rows:
        cells[row[key]][row["corruption"], row["severity"]].append(float(row["accuracy"]))
    return {name: sum(sum(values) / len(values) for values in table.values()) / len(table)
            for name, table in cells.items()}


def tentCurveAt(ladder, distance):
    """TENT's own accuracy at the same displacement, read off its ladder.

    ladder is (displacement, accuracy) pairs measured at one batch size. Linear in log
    displacement, which is the scale the ladder is spaced on. Returns None outside the
    ladder's range, where reading it off would be an extrapolation.
    """
    byDistance = defaultdict(list)
    for x, y in ladder:
        byDistance[x].append(y)
    points = sorted((x, sum(y) / len(y)) for x, y in byDistance.items())
    if len(points) < 2 or not points[0][0] <= distance <= points[-1][0]:
        return None
    for (left, leftAccuracy), (right, rightAccuracy) in zip(points, points[1:]):
        if left <= distance <= right:
            share = (math.log(distance) - math.log(left)) / (math.log(right) - math.log(left))
            return leftAccuracy + share * (rightAccuracy - leftAccuracy)
    return None


def mean(values):
    return sum(values) / len(values)


def std(values):
    """Population standard deviation, the same convention evaluation/metrics.meanStd uses."""
    if len(values) < 2:
        return 0.0
    centre = mean(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / len(values))


def ladderBySeed(rows):
    """{seed -> {variant -> {"accuracy", "displacement", "batch"}}}, over that seed's cells."""
    gathered = defaultdict(lambda: defaultdict(lambda: {"accuracy": [], "displacement": [],
                                                         "batch": None}))
    for row in rows:
        entry = gathered[row["seed"]][row["variant"]]
        entry["accuracy"].append(100 * float(row["accuracy"]))
        entry["displacement"].append(float(row["displacementNorm"]))
        entry["batch"] = row["batch"]
    return {seed: {name: {"accuracy": mean(values["accuracy"]),
                          "displacement": mean(values["displacement"]),
                          "batch": values["batch"],
                          "size": len(values["accuracy"])}
                   for name, values in variants.items()}
            for seed, variants in gathered.items()}


def pooledBySeed(bySeed):
    """One entry per variant, averaging the seeds that produced it, for the plot."""
    gathered = defaultdict(lambda: {"accuracy": [], "displacement": [], "batch": None})
    for variants in bySeed.values():
        for name, entry in variants.items():
            gathered[name]["accuracy"].append(entry["accuracy"])
            gathered[name]["displacement"].append(entry["displacement"])
            gathered[name]["batch"] = entry["batch"]
    return {name: {"accuracy": mean(values["accuracy"]),
                   "displacement": mean(values["displacement"]),
                   "batch": values["batch"]}
            for name, values in gathered.items()}


def matchedGaps(bySeed):
    """{variant -> [gap per seed]} against TENT's own curve at the same displacement.

    Each seed is scored on the ladder that seed measured, so the spread across seeds
    is the spread of the comparison itself rather than of either curve. A variant
    whose displacement falls outside its seed's ladder contributes no gap there, which
    is why a row can carry fewer seeds than the block ran.
    """
    gaps = defaultdict(list)
    # Sorted seed order, so the position a gap holds in its list names the seed it came from.
    for _seed, variants in sorted(bySeed.items()):
        ladder = defaultdict(list)
        for name, entry in variants.items():
            if name.startswith("tent") and entry["displacement"] > 0.0:
                ladder[entry["batch"]].append((entry["displacement"], entry["accuracy"]))
        for name, entry in variants.items():
            if name.startswith("tent") or entry["displacement"] <= 0.0:
                continue
            reference = tentCurveAt(ladder[entry["batch"]], entry["displacement"])
            if reference is not None:
                gaps[name].append(entry["accuracy"] - reference)
    return gaps


def stepSizeFigure():
    """Accuracy against the distance the parameters actually travel over one cell.

    The axis is the `displacementNorm` column every accuracy run records for itself,
    ||theta_after - theta_before|| over the 9,600 BN affine parameters, measured on the
    run rather than estimated from a gradient norm at the source point. The three ladders
    -- TENT's step size, the consistency term's lambda, and the batch size -- all move
    that distance, so if the points fall on one curve per batch size, the gain the
    ablation attributed to the consistency term is movement rather than a mechanism.

    Alongside the figure, each consistency point is scored against TENT's curve at the
    same displacement and the same batch size, which turns that question into a number.
    The ladder block runs that comparison at three seeds, so each gap is printed with its
    seed-to-seed spread: a mean gap the spread covers is not a detected effect.
    """
    rows = readRows("ladder.csv")
    if not rows:
        print("skipped stepSize: results/ladder.csv not written yet")
        return
    bySeed = ladderBySeed(rows)
    summary = pooledBySeed(bySeed)
    series = {"tent": [], "consistency": []}
    for name, entry in summary.items():
        # A zero displacement has no place on a log axis and no reading on the ladder.
        if entry["displacement"] <= 0.0:
            continue
        series["tent" if name.startswith("tent") else "consistency"].append(
            (entry["displacement"], entry["accuracy"], name, entry["batch"]))
    if len(series["tent"]) < 2 or len(series["consistency"]) < 2:
        print("skipped stepSize: results/ladder.csv holds fewer than two points per family")
        return

    distances = {name: entry["displacement"] for name, entry in summary.items()}
    accuracies = {name: entry["accuracy"] for name, entry in summary.items()}
    sizes = {name: entry["batch"] for name, entry in summary.items()}
    gaps = matchedGaps(bySeed)
    print("\n== matched displacement, read off the displacementNorm column of ladder.csv ==")
    print(f"  {'variant':26s}{'batch':>7s}{'distance/cell':>15s}{'acc':>8s}"
          f"{'gap vs TENT (pp)':>19s}{'spread':>9s}{'seeds':>7s}")
    for name in sorted(gaps, key=lambda name: (sizes[name], distances[name])):
        values = gaps[name]
        print(f"  {name:26s}{sizes[name]:>7s}{distances[name]:15.3f}{accuracies[name]:8.2f}"
              f"{mean(values):+14.2f}{std(values):9.2f}{len(values):>7d}")
    readable = {name: values for name, values in gaps.items() if sizes[name] == "64"}
    flat = [value for values in readable.values() for value in values]
    if flat:
        print(f"  batch 64, all {len(flat)} readings: "
              f"{mean(flat):+.2f} ± {std(flat):.2f} pp, "
              f"positive in {sum(1 for value in flat if value > 0)} of {len(flat)}")
        # One mean per seed. Within a seed every variant shares the batch order and the
        # augmentation draws, so these three numbers are the paired reading.
        print("  batch 64, mean gap by seed: " + "   ".join(
            f"seed {seed} {mean([values[index] for values in readable.values()
                                 if len(values) > index]):+.2f} pp"
            for index, seed in enumerate(sorted(bySeed))))

    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    markers = {"64": "o", "32": "s", "16": "^"}
    for family, colour, label in (("tent", "tab:blue", "TENT"),
                                  ("consistency", "tab:red", "TENT + consistency term")):
        bySize = defaultdict(list)
        for point in series[family]:
            bySize[point[3]].append(point)
        for size, points in sorted(bySize.items()):
            points = sorted(points)
            # Only the batch-64 ladders are joined: those two sweeps are the matched control.
            axis.plot([point[0] for point in points], [point[1] for point in points], color=colour,
                      marker=markers.get(size, "o"), linestyle="-" if size == "64" else "None",
                      linewidth=1.0, label=f"{label}, batch {size}")
            for distance, accuracy, name, _ in points:
                axis.annotate(name, (distance, accuracy), fontsize=6, color=colour,
                              textcoords="offset points", xytext=(4, 4))
    axis.set_xscale("log")
    axis.set_xlabel("||theta_after - theta_before|| per cell, measured on the run")
    axis.set_ylabel("mean accuracy over the validation cells (%)")
    axis.set_title("Both ladders on the measured displacement axis, one curve per batch size")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)
    writeFigure(figure, "stepSize",
                f"{len(series['tent'])} TENT and {len(series['consistency'])} consistency points, "
                f"{len(bySeed)} seed{'s' if len(bySeed) > 1 else ''}")


def ablationFigure():
    """Each gate configuration's mean gain over TENT, with how many cells moved up."""
    rows = readRows("ablation.csv")
    if not rows:
        print("skipped ablation: results/ablation.csv not written yet")
        return
    accuracies = cellAccuracy(rows)
    reference = accuracies["tent"]
    names = [name for name in accuracies if name != "tent"]
    gains = [100 * (accuracies[name] - reference) for name in names]
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.barh(names, gains, color=["tab:red" if gain > 0 else "tab:grey" for gain in gains])
    for index, gain in enumerate(gains):
        axis.annotate(f"{gain:+.2f} pp", (gain, index), fontsize=7,
                      textcoords="offset points", xytext=(4 if gain >= 0 else -34, 0))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel(f"mean accuracy gain over TENT (TENT = {reference * 100:.2f}%)")
    axis.set_title("gate ablation on 10 validation cells, one seed")
    figure.tight_layout()
    writeFigure(figure, "ablation", coverage(rows))


def severityFigure():
    """Accuracy against severity for the unseen corruptions, one line per method."""
    rows = readRows("mainTable.csv")
    if not rows:
        print("skipped severity curve: results/mainTable.csv not written yet")
        return
    unseen = set(loadConfig("splits")["test"]["unseenCorruption"]["corruptions"])
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["corruption"] in unseen:
            cells[row["method"]][int(row["severity"])].append(100 * float(row["accuracy"]))
    figure, axis = plt.subplots(figsize=(5.6, 4.0))
    for method in [name for name in METHOD_ORDER if name in cells]:
        severities = sorted(cells[method])
        axis.plot(severities, [sum(cells[method][s]) / len(cells[method][s]) for s in severities],
                  marker="o", label=method)
    axis.set_xticks(sorted({int(row["severity"]) for row in rows if row["corruption"] in unseen}))
    axis.set_xlabel("corruption severity (unseen corruption types)")
    axis.set_ylabel("accuracy (%)")
    axis.set_title("severity curves")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    writeFigure(figure, "severityCurves", coverage(rows))


def corruptionFigure():
    """One group of markers per corruption, all methods, from the main table."""
    rows = readRows("mainTable.csv")
    if not rows:
        print("skipped per-corruption figure: results/mainTable.csv not written yet")
        return
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cells[row["corruption"]][row["method"]].append(100 * float(row["accuracy"]))
    corruptions = sorted(cells)
    methods = [name for name in METHOD_ORDER if name in next(iter(cells.values()))]
    figure, axis = plt.subplots(figsize=(9.0, 4.0))
    width = 0.9 / len(methods)
    for index, method in enumerate(methods):
        positions = [position + index * width for position in range(len(corruptions))]
        axis.plot(positions, [sum(cells[c][method]) / len(cells[c][method]) for c in corruptions],
                  marker="o", linestyle="None", label=method)
    axis.set_xticks([position + 0.45 - width / 2 for position in range(len(corruptions))])
    axis.set_xticklabels(corruptions, rotation=60, ha="right", fontsize=7)
    axis.set_ylabel("accuracy (%)")
    # Markers rather than bars, because the methods span 59 to 87% and an axis from zero
    # would flatten the differences this figure exists to show. A truncated bar would
    # overstate them; a point at a height needs no such convention.
    axis.set_ylim(55, 92)
    axis.set_title("accuracy per corruption type, test cells and seeds averaged")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=7, ncols=6, loc="upper center", framealpha=0.95)
    writeFigure(figure, "perCorruption", coverage(rows))


def batchSweepFigure():
    """Stream accuracy against batch size, from results/stream.csv."""
    rows = readRows("stream.csv")
    if not rows:
        print("skipped batch sweep: results/stream.csv not written yet")
        return
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["segment"] == "all":
            cells[row["batch"]][row["method"]].append(100 * float(row["accuracy"]))
    figure, axis = plt.subplots(figsize=(5.6, 4.0))
    batches = sorted(cells, key=int)
    methods = {row["method"] for row in rows if row["segment"] == "all"}
    for method in [name for name in METHOD_ORDER if name in methods]:
        axis.plot([int(b) for b in batches],
                  [sum(cells[b][method]) / len(cells[b][method]) for b in batches],
                  marker="o", label=method)
    axis.set_xscale("log", base=2)
    axis.set_xticks([int(b) for b in batches])
    axis.set_xticklabels(batches)
    axis.set_xlabel("batch size (online stream, severity 5)")
    axis.set_ylabel("accuracy over the whole stream (%)")
    axis.set_title("batch-size sweep")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="upper left")
    writeFigure(figure, "batchSweep", coverage(rows))


def main(args=()):
    for figureBuilder in (stepSizeFigure, ablationFigure, severityFigure,
                          corruptionFigure, batchSweepFigure):
        figureBuilder()
