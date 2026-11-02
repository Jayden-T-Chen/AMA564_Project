"""Figures for the report, drawn from the CSVs under results/.

Run:  .venv/bin/python -m run.makeFigures

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
    return list(csv.DictReader(open(path))) if path.exists() else None


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


def tentCurveAt(ladder, path):
    """TENT's own accuracy at the same displacement, read off its learning-rate ladder.

    ladder is (path, accuracy) pairs measured at one batch size. Linear in log path, which
    is the scale the ladder is spaced on. Returns None outside the ladder's range, where
    reading it off would be an extrapolation.
    """
    byPath = defaultdict(list)
    for x, y in ladder:
        byPath[x].append(y)
    points = sorted((x, sum(y) / len(y)) for x, y in byPath.items())
    if len(points) < 2 or not points[0][0] <= path <= points[-1][0]:
        return None
    for (left, leftAccuracy), (right, rightAccuracy) in zip(points, points[1:]):
        if left <= path <= right:
            share = (math.log(path) - math.log(left)) / (math.log(right) - math.log(left))
            return leftAccuracy + share * (rightAccuracy - leftAccuracy)
    return None


def stepSizeFigure():
    """Accuracy against the distance the parameters travel over one cell.

    The ladders this repo ran live on different axes: TENT's learning rate, the
    consistency term's lambda, and the batch size. cellPathLength = lr * ||grad L|| *
    batches per cell is the one axis all three share, so if every point falls on one
    curve, the gain the ablation attributed to the consistency term is movement rather
    than a different mechanism.

    Alongside the figure, each consistency point is scored against TENT's curve at the
    same displacement and the same batch size, which turns that question into a number.

    A missing gradientScale-<block>.csv only costs that block's points, because the
    accuracy runs and the gradient probe are separate commands.
    """
    series = {"tent": [], "consistency": []}
    ladder = defaultdict(list)
    seen = set()
    for block in ("stepControl", "smallBatch", "batch16Tuning", "probe"):
        scales = readRows(f"gradientScale-{block}.csv")
        accuracies = readRows(f"{block}.csv")
        if not scales or not accuracies:
            continue
        byVariant = cellAccuracy(accuracies)
        for row in scales:
            path = float(row["cellPathLength"])
            family = "tent" if row["variant"].startswith("tent") else "consistency"
            point = (path, 100 * byVariant[row["variant"]], row["variant"], row["batchSize"],
                     float(row["lambdaWeight"]))
            # The blocks overlap: smallBatch re-ran the batch-64 lambda 4 configuration that
            # stepControl already measured, and got the same accuracy. Keep one copy.
            if (family, point[3], point[4], path) not in seen:
                seen.add((family, point[3], point[4], path))
                series[family].append(point)
                if family == "tent":
                    ladder[point[3]].append((path, point[1]))
    if len(series["tent"]) < 2 or len(series["consistency"]) < 2:
        print("skipped stepSize: need gradientScale-<block>.csv next to <block>.csv for both ladders")
        return

    # The matched comparison: TENT's accuracy at the same batch size and the same total
    # displacement. A positive gap means the consistency term beat plain movement.
    print("  lambda  batch  path/cell  ours   TENT at the same displacement   gap (pp)")
    for path, accuracy, _, size, weight in sorted(series["consistency"]):
        reference = tentCurveAt(ladder[size], path)
        score = f"{reference:7.2f}  {accuracy - reference:+6.2f}" if reference else "   --       --"
        print(f"    {weight:>4}  {size:>5}  {path:9.3f}  {accuracy:6.2f}  {score}")

    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    markers = {"64": "o", "32": "s", "16": "^", "256": "P"}
    for family, colour, label in (("tent", "tab:blue", "TENT"),
                                  ("consistency", "tab:red", "TENT + consistency term")):
        bySize = defaultdict(list)
        for point in series[family]:
            bySize[point[3]].append(point)
        for size, points in sorted(bySize.items()):
            points = sorted(points)
            # Only the batch-64 ladders are joined: those two sweeps are the matched control.
            axis.plot([point[0] for point in points], [point[1] for point in points], color=colour,
                      marker=markers[size], linestyle="-" if size == "64" else "None",
                      linewidth=1.0, label=f"{label}, batch {size}")
            for path, accuracy, name, _, _ in points:
                axis.annotate(name.replace("_", " "), (path, accuracy), fontsize=6, color=colour,
                              textcoords="offset points", xytext=(4, 4))
    axis.set_xscale("log")
    axis.set_xlabel("parameter distance travelled per cell, lr * |grad L| * batches")
    axis.set_ylabel("mean accuracy over the validation cells (%)")
    axis.set_title("Both ladders on one axis: does the consistency term move along TENT's curve?")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)
    writeFigure(figure, "stepSize",
                f"{len(series['tent'])} TENT and {len(series['consistency'])} consistency points")


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
    """One bar group per corruption, all methods, from the main table."""
    rows = readRows("mainTable.csv")
    if not rows:
        print("skipped per-corruption bars: results/mainTable.csv not written yet")
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
        axis.bar(positions, [sum(cells[c][method]) / len(cells[c][method]) for c in corruptions],
                 width, label=method)
    axis.set_xticks([position + 0.45 - width / 2 for position in range(len(corruptions))])
    axis.set_xticklabels(corruptions, rotation=60, ha="right", fontsize=7)
    axis.set_ylabel("accuracy (%)")
    axis.set_title("accuracy per corruption type, test cells and seeds averaged")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8, ncols=2)
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
    axis.set_xscale("log2")
    axis.set_xticks([int(b) for b in batches])
    axis.set_xlabel("batch size (online stream, severity 5)")
    axis.set_ylabel("accuracy over the whole stream")
    axis.set_title("batch-size sweep")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    writeFigure(figure, "batchSweep", coverage(rows))


def main():
    for figureBuilder in (stepSizeFigure, ablationFigure, severityFigure,
                          corruptionFigure, batchSweepFigure):
        figureBuilder()


if __name__ == "__main__":
    main()
