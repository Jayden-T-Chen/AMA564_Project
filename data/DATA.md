# Data provenance

Everything below was measured on this machine on 2026-11-02, not copied from a README.

## CIFAR-10-C

| item | value |
|---|---|
| file | `data/raw/CIFAR-10-C.tar` |
| bytes | 2,918,471,680 |
| sha256 | `c72763e101c723b7c507b96205f7e938912a5d587376173b825850cf3cb876a7` |
| canonical source | Zenodo record 2535967, DOI 10.5281/zenodo.2535967 ("CIFAR-10-C and CIFAR-10-P") |
| mirror actually used | `https://hf-mirror.com/datasets/Sijiax/CIFAR-10-C/resolve/main/CIFAR-10-C.tar` (identical byte length) |
| measured throughput | mirror 7.4 MB/s (20 MB range sample); direct Zenodo 283 KB/s |
| extracted | `data/CIFAR-10-C/`, 20 `.npy` files, 2.7 GB |
| benchmark paper | Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (corruption benchmark); reused data, cited in the report |

Archive layout, verified by reading the arrays:

- 19 corruption arrays, each `(50000, 32, 32, 3)` uint8 = five severity blocks of
  **10,000 images**, severity 1 first. Total for the official 15 types: 750,000 images.
- `labels.npy` is `(50000,)` uint8: the 10,000 CIFAR-10 test labels tiled five times,
  so a row index addresses images and labels with the same slice.
- Severity order confirmed from the data: `brightness` block means rise
  131.6 → 141.8 → 151.6 → 160.9 → 177.5, and `gaussian_noise` block std rises
  64.55 → 65.21 → 66.11 → 66.63 → 67.21.

**Scope note.** The archive ships four corruption types beyond the original 15 —
`speckle_noise`, `gaussian_blur`, `spatter`, `saturate` — from the C+P extension.
This study uses only the official 15 listed in `configs/splits.yaml`, which keeps the
750,000-image count and the 15 x 5 grid stated in the proposal.

## CIFAR-10

Downloaded by `torchvision` into `data/cifar10/` from
`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` (a direct HEAD request from this
network returned 200 in 4.7 s). 50,000 training and 10,000 test images, 10 classes.
Normalisation statistics are fit on the training split only.

## Memory note

Full CIFAR-10-C pixel data is 2.3 GB as uint8, which does not fit alongside PyTorch in
8 GB of unified memory. Loaders therefore `np.load(..., mmap_mode="r")` one corruption at
a time and copy only the block being used.
