"""The study's one entry point: read the tables, check them live, or run a stage.

Run:  .venv/bin/python -m run.runAll                    # reads results/*.csv only
      .venv/bin/python -m run.runAll check              # plus two live checks
      .venv/bin/python -m run.runAll gate ladder        # one stage, with its own argument
      .venv/bin/python -m run.runAll test               # the unit tests, no data needed

With no argument this prints the frozen configuration, the six-method main table, the
stream sweep and the matched-displacement table, and redraws the five figures, all from
the committed CSVs, so a reader can confirm that the report's tables and its pictures come
from the same rows on any machine. `check` adds two live checks: what the label mask does
to a single update, which is the fact `oracle` is reported as, and the real code over the
ten validation cells against the committed rows for the same configuration, which is the
check that this code still produces the numbers those rows hold. The remaining commands
are the stages that wrote those rows: each is one call into the module that owns it, so
the cell lists, the CSV paths and the printing stay defined in exactly one place and no
stage has a command line of its own.

Nothing here decides anything: every table comes from the same functions the runs that
produced them used, so this file can drift only by going wrong, not by disagreeing.
"""

import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import torch

from configIO import loadConfig
from dataset.cifar10CLoader import cell as cellImages
from dataset.preprocess import prepare
from evaluation.metrics import meanStd
from evaluation.protocols import cellList, evaluateCells
from methods.tent import makeTentPredict, makeTentStep
from models.resNet18 import loadSourceModel
from run import (probeThroughput, runAblation, runGradientScale, runMain, runPilot,
                 runStream, trainSource)
from run.makeFigures import METHOD_ORDER, main as writeFigures, resultPath
from run.runMain import printMeanTable, samplesFromRows
from run.runPilot import methodFactories as pilotFactories
from run.runUtils import batchOrder, readRows, resolveDevice, setSeed
from run.trialDriver import buildContext, reloadModel

def runTests(args=()):
    """The unit tests: CPU tensors, no data, no checkpoint, nothing written.

    An optional argument narrows the discovery pattern, e.g. `test testGateWeights.py`.
    """
    root = Path(__file__).resolve().parent.parent
    pattern = args[0] if args else "test*.py"
    suite = unittest.TestLoader().discover(str(root / "tests"), pattern=pattern,
                                           top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


# The stages that produced results/*.csv, in the order the study runs them, each with the
# line `runAll` prints for an unknown or missing command. `test` is the one stage that
# writes nothing.
STAGES = {
    "train": (trainSource.main, "train the source model to freeze [--epochs N]"),
    "throughput": (probeThroughput.main, "measure the three step types on this device"),
    "pilot": (runPilot.main, "pilot gate on the 10 validation cells -> pilot.csv"),
    "gate": (runAblation.main, "one variant block of configs/gate.yaml [ablation]"),
    "gradient": (runGradientScale.main, "gradient norms of one block [ladder]"),
    "main": (runMain.main, "the 65-cell grid, six methods x three seeds -> mainTable.csv"),
    "stream": (runStream.main, "the online stream [sweep] -> stream.csv"),
    "figures": (writeFigures, "redraw the five figures from the CSVs"),
    "test": (runTests, "unit tests on CPU tensors [module pattern], no data needed"),
}

# The methods with a committed validation-cell pass to be checked against, each with the
# CSV, the column and the value naming it. pilot.csv holds the three baselines; the
# tentTuned row is TENT at adaptation.tunedLr on the same cells, which is one ladder.csv
# variant whose name is derived from that number rather than written next to it, so the
# two cannot drift apart when the ladder selects a different step size. The consistency
# term is not compared because no committed block runs it at the configs/gate.yaml
# defaults -- every ablation row pins its own switches.
COMMITTED = {"sourceOnly": ("pilot.csv", "method", "sourceOnly"),
             "bnAdapt": ("pilot.csv", "method", "bnAdapt"),
             "tent": ("pilot.csv", "method", "tent")}


def committedRows(adaptation):
    """{method -> (csv, column, value)} for the passes a live check can be read against."""
    multiples = round(adaptation["tunedLr"] / adaptation["lr"])
    return dict(COMMITTED, tentTuned=("ladder.csv", "variant", f"tent64_lr{int(multiples)}"))


def printSetup():
    """The frozen decisions on one screen, with the file each one lives in."""
    base, splits, gate = loadConfig("base"), loadConfig("splits"), loadConfig("gate")
    adaptation, validation = base["adaptation"], splits["validation"]
    trained = json.loads(resultPath("sourceTrain.json").read_text())
    epochs = trained["curve"]
    print("== what is frozen ==")
    print(f"source model      {trained['checkpoint']}, {len(epochs)} epochs, "
          f"clean test {epochs[-1]['testAccuracy'] * 100:.2f}% in "
          f"{sum(step['seconds'] for step in epochs) / 60:.1f} min")
    print(f"device            {resolveDevice()}")
    print(f"adaptation        BN {adaptation['bnMode']}-mode, batch {adaptation['batchSize']}, "
          f"lr {adaptation['lr']} (tentTuned {adaptation['tunedLr']}), "
          f"{adaptation['stepsPerBatch']} step per batch, seeds {adaptation['seeds']}, "
          f"static cells {'episodic' if adaptation['resetPerCell'] else 'chained'}")
    consistency = gate["consistencyGate"]
    print(f"final objective   entropy gate use={str(gate['entropyGate']['use']).lower()}, "
          f"consistency gate weight {consistency['weight']}, "
          f"inLoss={str(consistency['inLoss']).lower()}, "
          f"inWeights={str(consistency['inWeights']).lower()} (configs/gate.yaml defaults)")
    print(f"cells             {len(validation['corruptions'])} corruptions x "
          f"{validation['severities']} at {validation['imagesPerCell']} images for tuning, "
          f"{len(splits['test']['seenCorruption']['corruptions'])} seen x "
          f"{splits['test']['seenCorruption']['severities']} plus "
          f"{len(splits['test']['unseenCorruption']['corruptions'])} unseen x "
          f"{splits['test']['unseenCorruption']['severities']} for testing")


def printMainTable():
    """The 65-cell table, from the same function runMain prints with."""
    printMeanTable(samplesFromRows(readRows(resultPath("mainTable.csv"))))


def printStreamTable():
    """Stream accuracy, worst segment and travelled distance per batch size."""
    rows = readRows(resultPath("stream.csv"))
    whole, segmentMinimum, travelled = defaultdict(list), {}, defaultdict(list)
    for row in rows:
        key = (int(row["batch"]), row["method"])
        if row["segment"] == "all":
            whole[key].append(float(row["accuracy"]))
            travelled[key].append(float(row.get("displacementNorm") or 0.0))
        else:
            seeded = (key, row["seed"])
            segmentMinimum[seeded] = min(segmentMinimum.get(seeded, 1.0), float(row["accuracy"]))
    worst = defaultdict(list)
    for (key, _seed), value in segmentMinimum.items():
        worst[key].append(value)
    print("\n== online stream, mean over seeds, the worst segment and the distance travelled ==")
    print(f"{'batch':6s}{'method':16s}{'stream acc':>12s}{'std':>7s}{'worst segment':>15s}"
          f"{'path/stream':>13s}")
    for batch in sorted({key[0] for key in whole}):
        for method in [name for name in METHOD_ORDER if (batch, name) in whole]:
            mean, std = meanStd(whole[(batch, method)])
            path = sum(travelled[(batch, method)]) / len(travelled[(batch, method)])
            print(f"{batch:<6d}{method:16s}{mean * 100:11.2f}%{std * 100:6.2f}"
                  f"{meanStd(worst[(batch, method)])[0] * 100:14.2f}%{path:13.3f}")


def oracleEquivalence(context):
    """What the label mask does to one update, measured rather than asserted.

    `oracle` is reported as a reference point, so the two facts that fix it are checked
    here: a mask that keeps every sample reproduces plain TENT's parameter update bit for
    bit, and a mask that keeps none leaves the parameters exactly where they were. Both
    are read on the affine parameters a method may move. The BatchNorm running statistics
    are excluded, because every train-mode forward rewrites them and no method reads them.
    """
    corruption, severity = context["cells"][0]
    images, _ = cellImages(context["root"], corruption, severity, context["batchSize"])
    batch = prepare(images, context["meanStd"], context["device"])
    # Four models are alive at once here, so this is the one place the single shared
    # object of reloadModel cannot serve: the three updates would land on the same
    # weights. A fourth model reads the labels, so each of the three compared models runs
    # exactly one forward and their running statistics stay comparable.
    fresh = lambda: loadSourceModel(context["checkpointPath"], context["device"])[0]
    reader = fresh()
    reader.train()
    with torch.no_grad():
        predictions = reader(batch).argmax(dim=1)

    models = {name: fresh() for name in ("plain", "keepAll", "keepNone")}
    affine = lambda model: {key: parameter.detach().clone()
                            for key, parameter in model.named_parameters()
                            if key.endswith(("weight", "bias"))}
    before = affine(models["plain"])
    targets = {"plain": None, "keepAll": predictions, "keepNone": (predictions + 1) % 10}
    moved = {}
    for name, model in models.items():
        step = makeTentStep(model, context["adaptation"]["lr"], labelMask=name != "plain")
        setSeed(context["seed"])
        step(batch, targets[name])
        moved[name] = affine(model)

    furthest = lambda left, right: max(float((left[key] - right[key]).abs().max())
                                       for key in left)
    print(f"\n== oracle equivalence, one batch of {corruption} severity {severity} ==")
    print(f"  plain TENT against its own start      {furthest(moved['plain'], before):.3e}"
          "     an update happened")
    print(f"  keep-none mask against its own start  {furthest(moved['keepNone'], before):.3e}"
          "     no parameter moved")
    print(f"  keep-all mask against plain TENT      "
          f"{furthest(moved['keepAll'], moved['plain']):.3e}     the same update")
    print(f"  running statistics, plain against keep-all  "
          f"{furthest(dict(models['plain'].named_buffers()), dict(models['keepAll'].named_buffers())):.3e}"
          "     both saw the identical batch")


def spotCheck(context):
    """The validation cells, live, against the committed rows for the same configuration.

    The cells follow adaptation.resetPerCell, the same switch `run.runAll pilot`
    and `run.runAll gate` wrote their rows under, so this check compares like with
    like: under the episodic default every cell starts from the frozen checkpoint,
    and flipping the switch back to chained reproduces the rows a chained run wrote.
    """
    adaptation, cells = context["adaptation"], context["cells"]
    factories = dict(pilotFactories(adaptation["lr"]))
    factories["tentTuned"] = lambda model: makeTentPredict(model, adaptation["tunedLr"])
    # Only the committed rows this seed produced. ladder.csv now carries three, and
    # reading the variant name alone would let the last seed win.
    committed, missing = {}, []
    for name, (csvName, column, rowValue) in committedRows(adaptation).items():
        rows = [row for row in readRows(resultPath(csvName)) or []
                if row[column] == rowValue and int(row["seed"]) == context["seed"]]
        for row in rows:
            committed[name, row["corruption"], int(row["severity"])] = float(row["accuracy"])
        if not rows:
            missing.append(f"{name}: no {csvName} row {rowValue} at seed {context['seed']}")
    print(f"\n== live check on the {len(cells)} validation cells, batch {context['batchSize']}, "
          f"checkpoint {context['checkpointPath'].name}, "
          f"{'episodic' if context['resetPerCell'] else 'chained'} ==")
    for note in missing:
        print(f"  skipped {note}")
    print(f"{'method':16s}{'mean acc':>10s}{'cells checked':>15s}{'max |delta|':>13s}")
    mismatches = []
    checked = 0
    for name, makePredict in factories.items():
        if (name, *cells[0]) not in committed:
            continue
        setSeed(context["seed"])
        table = evaluateCells(context["root"], cells, context["count"], context["meanStd"],
                              context["device"], context["batchSize"], context["seed"],
                              makePredict, lambda: reloadModel(context),
                              context["resetPerCell"])
        deltas = [(cell, (table[cell]["accuracy"] - committed[name, *cell]) * 100)
                  for cell in cells]
        mismatches += [(name, cell, delta) for cell, delta in deltas if abs(delta) > 0.001]
        mean = sum(result["accuracy"] for result in table.values()) / len(table)
        checked += len(cells)
        print(f"{name:16s}{mean * 100:9.2f}%{len(cells):14d}"
              f"{max(abs(delta) for _, delta in deltas):12.4f} pp", flush=True)
    for name, cell, delta in mismatches:
        print(f"  mismatch {name:14s} {cell[0]} sev{cell[1]}: {delta:+.4f} pp")
    print(f"{checked - len(mismatches)} of {checked} live accuracies equal their committed rows "
          f"to the stored precision")


def report():
    """Every table and every figure, from the committed CSVs."""
    printSetup()
    printMainTable()
    printStreamTable()
    writeFigures()


def check():
    """The report, then the two live checks against it."""
    report()
    validation = loadConfig("splits")["validation"]
    # Nothing is appended to this CSV: both checks print, they do not record.
    context = buildContext(cellList(validation), validation["imagesPerCell"], "runAllCheck.csv")
    oracleEquivalence(context)
    spotCheck(context)


def printUsage():
    print("usage:  python -m run.runAll [command] [argument]")
    print(f"        {'report':11s} every table and figure, from results/*.csv alone")
    print(f"        {'check':11s} report, plus the oracle equivalence and the ten "
          f"validation cells run live")
    for name, (_stage, description) in STAGES.items():
        print(f"        {name:11s} {description}")


def main():
    command, argument = (sys.argv[1], tuple(sys.argv[2:])) if len(sys.argv) > 1 else ("", ())
    if command in ("", "report"):
        report()
    elif command == "check":
        check()
    elif command in STAGES:
        STAGES[command][0](argument)
    else:
        printUsage()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
