# AMA 564 Course Project — Gated TENT under mixed online distribution shift

Team Tim: Chen Xiyan (26080997g), OuYang Jiaying (26076982g), Zhang Zhenshuo (26127097g),
Liu Fangze (26119048g).

**Question.** Does adding a sample-selection gate (entropy threshold plus
augmentation-consistency reweighting) to TENT improve mean and worst-case corruption
accuracy under a mixed, online distribution shift, and at what cost?

## Layout

| path | holds |
|---|---|
| `configs/` | the frozen decisions: `base.yaml` (run settings), `splits.yaml` (which cells tune and which are evaluated), `gate.yaml` (gate design and sweeps) |
| `dataset/` | reading CIFAR-10 / CIFAR-10-C, normalisation, online stream construction |
| `models/` | ResNet-18 and the BN-affine-only interface every method shares |
| `methods/` | loss terms, TENT, and the gated variant |
| `evaluation/` | metrics, protocol drivers, latency and memory |
| `run/` | one entry point per experiment |
| `results/` | committed CSV tables and figures |
| `data/` | raw archive (gitignored) plus `DATA.md` provenance |

## Setup

```bash
cd "AMA564 Project"
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.lock.txt
```

## Data

```bash
curl -L -C - -o data/raw/CIFAR-10-C.tar \
  https://hf-mirror.com/datasets/Sijiax/CIFAR-10-C/resolve/main/CIFAR-10-C.tar
shasum -a 256 data/raw/CIFAR-10-C.tar   # expect c72763e1...cb876a7, 2918471680 bytes
tar -xf data/raw/CIFAR-10-C.tar -C data/
```

CIFAR-10 downloads itself on first use of `dataset/cifar10Source.py`.
Provenance, array shapes and the severity-block check are recorded in `data/DATA.md`.

## Reproducing

```bash
.venv/bin/python -m run.probeThroughput   # machine capability, no checkpoint needed
.venv/bin/python -m run.trainSource       # one frozen ResNet-18 checkpoint
.venv/bin/python -m run.runPilot          # baselines on the validation cells, gate verdict
.venv/bin/python -m run.runTuning         # keepFraction then lambda, on the validation cells
.venv/bin/python -m run.runAblation               # results/ablation.csv, the gate roles split
.venv/bin/python -m run.runAblation probe         # results/probe.csv
.venv/bin/python -m run.runAblation stepControl   # results/stepControl.csv, lambda vs lr
.venv/bin/python -m run.runAblation smallBatch    # results/smallBatch.csv, batch 16 / 32
.venv/bin/python -m run.runAblation batch16Tuning  # results/batch16Tuning.csv, TENT's lr ladder at batch 16
.venv/bin/python -m run.runGradientScale stepControl   # results/gradientScale-stepControl.csv, ||grad|| per ladder
.venv/bin/python -m run.runGradientScale smallBatch    # results/gradientScale-smallBatch.csv
.venv/bin/python -m run.runMain                   # results/mainTable.csv, 65 cells x 6 methods x 3 seeds
.venv/bin/python -m run.runMain oracle            # just the oracle rows, appended to the same CSV
.venv/bin/python -m run.runStream                 # results/stream.csv, online stream at batch 64
.venv/bin/python -m run.runStream sweep           # results/stream.csv, batch 16 / 64 / 256
.venv/bin/python -m run.makeFigures               # results/figures/*.png, plus the matched-displacement table
```

Each entry point prints its own summary and appends rows to one CSV under `results/`,
with the seed, cell, method and effective config on every row, so a number can be traced
back to one command. `run/runAblation.py` reads a named variant block from
`configs/gate.yaml`; a variant may override `lr` or `batch`, which is how the probe blocks
change one quantity at a time. Seeds and batch size come from `configs/base.yaml`. A run
replaces its own CSV from the first row, so an interrupted block leaves nothing behind —
always pass the block name, since the default is the full `ablation` block.
`run/runGradientScale.py` measures the gradient norm of the same variants the accuracy
blocks ran, and `run/makeFigures.py` joins the two into `cellPathLength`, the axis both
ladders share.

## Source model

`models/checkpoints/resNet18Seed7.pt`: ResNet-18 from scratch on CIFAR-10, 100 epochs,
clean test accuracy **88.07%**. It is below the ~92% a CIFAR-tuned ResNet-18 reaches
because torchvision's variant keeps the ImageNet 7x7 stride-2 stem plus maxpool instead
of a 3x3 stem; every method reads the same frozen checkpoint, so the comparison is
unaffected and the report lists this as a limitation. The checkpoint also carries the
normalisation statistics fit on the training split, which is what all later runs use.

## Protocol notes

- Accuracy is read **after** the update, on the same batch — the posterior evaluation of
  the original TENT code: one gradient step, then classify the batch with the updated
  affine parameters. This costs one extra forward pass per batch over a prior readout.
- Batches are always full. `run/runUtils.batchOrder` permutes a cell's indices and
  repeats the head of the permutation so the last batch is not short, so a 5,000-image
  cell is scored on 5,056 images with 56 counted twice. Every method at the same seed
  sees the identical sequence, which is what makes the comparison paired.
- BN-adapt is what separates the two effects a reader might otherwise conflate: it is
  batch-statistics recalibration alone, with the weights untouched. A gain of TENT over
  source-only that does not also appear against BN-adapt is recalibration, not the
  entropy objective, and `run/runPilot.py` prints both columns.
- Tuning budgets are matched. The gate received six validation-cell trials, three
  `keepFraction` values and three `lambda` values (`tuning` in `configs/gate.yaml`). Once
  the ablation showed the consistency term's gain tracking the step size, TENT received
  seven trials over lr 0.0005 to 0.032 (`results/probe.csv`, `results/stepControl.csv`),
  and the main table carries both TENT at the shared lr and `tentTuned` at the lr those
  trials selected. The guidance asks for comparable tuning budgets, so this is stated
  rather than left to the reader.
- `oracle` is TENT with the entropy loss restricted to the samples it already classifies
  correctly before the update. It reads labels, so it is a ceiling rather than a method:
  it answers what any per-sample selection rule could reach, which is the question the
  two gates were meant to answer cheaply. It is the same objective with a 0-1 mask
  (`labelMask` in `methods/tent.py`), so it costs no extra forward pass. Two checks pin
  it down: an all-correct mask reproduces plain TENT's parameter update to the last bit,
  and an all-wrong mask moves no parameter at all.

## What this machine measured (M1 Air, 8 GB, macOS 14.8.9, MPS)

| quantity | value |
|---|---|
| forward-only pass | 4,932 images/s (12.98 ms at batch 64) |
| one TENT step (forward + backward + SGD) | 2,176 images/s (29.41 ms at batch 64) |
| one consistency step (two forwards + one backward + SGD) | 1,155 images/s (55.43 ms at batch 64) |
| peak allocated memory | 67 MB for all three, the augmented view adds no ceiling |
| BN affine parameters updated | 9,600 out of 11,181,642 |
| source training, one epoch of 50,000 images at batch 128 | 39.4 s mean (43.9 s first epoch, 30.7-46.5 s range), whole 100-epoch run 65.7 min |
| one 5,056-image validation cell, end to end | source-only 1.3 s, BN-adapt 1.4 s, TENT 4.3 s, gated-TENT 7.0 s |
| one 10-cell validation pass, posterior readout included | 29-37 s for TENT, 53-64 s with the consistency term (`seconds` in `results/stepControl.csv`) |

Those end-to-end rates put the 65-cell test grid (10,080 images per cell) at 2.8 / 2.8 /
7.8 / 7.8 / 13.0 min for the five methods, so one seed of the whole main table is about
34 min and all three seeds about 103 min. The grid and the ablation therefore run on this
machine, and Google Colab T4 is a fallback rather than a requirement. The efficiency
answer to Q3 is in the two rows above: 1.9x TENT's per-batch latency for the augmented
view, and the same 67 MB peak.

## Environment notes

- **torch is pinned to 2.10.0.** torch 2.15.0 aborts on this machine: its MPS `linear`
  kernel calls an Objective-C API macOS 14.8.9 does not provide.
- **Batches must reach the network contiguous.** A train-mode BatchNorm backward on MPS
  crashes the process on permuted-stride input, which is why `dataset/preprocess.py`
  ends with `.contiguous()`.
- CIFAR-10-C is read with `mmap_mode="r"` one corruption at a time: all 750,000 images are
  2.3 GB uint8 and do not fit in 8 GB unified memory next to PyTorch.
- MPS has no deterministic-kernel switch, so the three seeds vary stream order and
  augmentation draws; the source checkpoint is a single frozen model. The report states
  this rather than implying full run-to-run determinism.
- On macOS framework Python, point the trust store at certifi before the first CIFAR-10
  download: `export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")`.
  Windows and CUDA machines do not need this.
