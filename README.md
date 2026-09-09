# LANCE — Low-Rank Activation Compression

> 🎉 **Accepted at ECCV 2026.**

Reference implementation of **LANCE** (Low-rank Activation Compression), from the
paper *"LANCE: Low Rank Activation Compression for Efficient On-Device Continual
Learning"* (M. P. E. Apolinario & K. Roy),
[arXiv:2509.21617](https://arxiv.org/abs/2509.21617).

Training's dominant memory cost on edge devices is storing intermediate
**activations** for the backward pass — often 5–10× the weight memory. LANCE
attacks this directly: it runs a **one-shot higher-order SVD (HOSVD)** over a few
calibration batches to obtain a fixed, reusable low-rank subspace per layer, then
during fine-tuning stores only the compact **core tensor** of each activation
(projected onto that subspace) instead of the full tensor. Because the subspace is
computed **once** and reused across all epochs, LANCE avoids the per-step
re-decomposition of prior low-rank methods, cutting both memory and
compute while keeping the forward pass exact.

Reported results: **25×–380×** activation-memory reduction (architecture
dependent) while staying within ~2% of full back-propagation on standard residual
networks across CIFAR-10/100, Oxford-IIIT Pets, Flowers-102, and CUB-200.

> This top level covers the **single-task fine-tuning** experiments. The
> **continual-learning** experiments (Split CIFAR-100 / Split MiniImageNet /
> 5-Datasets) live in [`continual_learning/`](continual_learning/), which has its
> own README, requirements, and demo script.

A single entry point, `main.py`, handles all backbones. The architecture selected
with `--arch` chooses the compression path automatically:

| `--arch`                                   | Path             | Compressed layer type |
|--------------------------------------------|------------------|-----------------------|
| `mobilenetv2`, `resnet18`, `resnet34`, `mcunet` | conv         | `Conv2d` (LANCE / HOSVD) |
| `deit_tiny` (timm)                         | linear           | `Linear` (LANCE / HOSVD) |

Each run selects one method: `--use-lance`, `--use-hosvd`, or none for the vanilla
back-propagation (BP) baseline.

## How LANCE works

1. **Phase I — one-shot subspace calibration (offline).** The last *N* trainable
   layers are wrapped in LANCE modules. For `--lance-warmup-batches` *N* mini-batches
   the pretrained model runs forward-only, accumulating a per-mode activation
   covariance. A single SVD per mode then yields orthonormal factor matrices
   `U_i`, truncated to the smallest rank capturing an energy fraction
   `--lance-threshold` (ε). This happens once, before training.
2. **Phase II — fine-tuning with low-rank activations.** During the forward pass
   each activation `X` is projected to a compact core `G = X ×(U_1ᵀ,…,U_dᵀ)` and
   **only `G` (plus the fixed factors) is stored** for backprop, shrinking the
   footprint from `∏ n_i` to `∏ r_i + Σ n_i r_i`. The backward pass reconstructs
   the weight gradient from `G` and the factors. Since the `U_i` are orthonormal,
   the LANCE weight gradient is an orthogonal projection of the true gradient —
   a valid descent direction (see the convergence analysis in the paper).

The energy threshold **ε (`--lance-threshold`)** is the main knob: higher ε keeps
more rank (higher accuracy, more memory); lower ε compresses harder. The paper uses
ε = 0.7 as a good accuracy/memory balance.

## Installation

```bash
pip install -r requirements.txt
```

### MCUNet (optional — only for `main.py --arch mcunet`)

MCUNet is not available on PyPI in the version used here. Install it from source
with a pinned version:

```bash
git clone https://github.com/mit-han-lab/mcunet.git mcunet_src
cd mcunet_src
# In setup.py, replace the dynamic version line with a fixed one:
#   VERSION = "0.1.1.post202508290149"
pip install .
cd .. && rm -rf mcunet_src
```

## Experiment logging (Weights & Biases)

The script logs metrics to **wandb**. The `--experiment-name` argument is used as
the wandb **project** name; run parameters are logged via `config`.

```bash
wandb login                 # first time only
# or run without an account:
export WANDB_MODE=offline
```

## Demos

Two scripts illustrate the method end to end. Datasets
download on first use; override `EPOCHS`, `LR`, `THRESHOLD`, `SEED`, etc. via
environment variables.

```bash
# ResNet18/34 on CIFAR-10/100 — BP baseline vs LANCE (last 2 and 4 conv layers)
DATA_DIR=~/Datasets bash demo_resnet.sh
EPOCHS=5 bash demo_resnet.sh              # fast smoke run

# DeiT-Tiny on CIFAR-100 — BP / LANCE / iso-memory HOSVD
DATA_DIR=~/Datasets bash demo_deit.sh
```

## Running

The paper's single-task fine-tuning recipe is **SGD, lr 0.05, batch 128, 50 epochs,
N = 100 calibration batches (`--lance-warmup-batches`), energy threshold ε = 0.7
(`--lance-threshold`)**, fine-tuning the last **2 or 4** layers of an
ImageNet-pretrained backbone. A single run:

```bash
python main.py --arch resnet18 --dataset cifar100 --data-dir ~/Datasets \
    --n-unfrozen 4 --epochs 50 --optim sgd --lr 0.05 --batch-size 128 \
    --use-lance --lance-threshold 0.7 --experiment-name resnet18_cifar100
```

Drop `--use-lance` for the BP baseline, or pass `--use-hosvd --explained-var 0.7`
for the iterative-HOSVD baseline (at most one method per run; the classifier head is
always trainable).

### DeiT-Tiny — Vision Transformer (`--arch deit_tiny`)

> These transformer experiments are **not** part of the final published paper; they
> were added during the rebuttal to show LANCE generalizes to attention
> architectures, with a **matched-memory** comparison against HOSVD
> last *N* `Linear` layers. `demo_deit.sh` covers **BP, LANCE, and iso-memory HOSVD**: LANCE 
> is calibrated first and saves its per-layer ranks (`--save-lance-ranks`); 
> HOSVD loads them (`--load-lance-ranks`, `deit_tiny` only) so both use the same activation budget.

**Matched-memory results on DeiT-Tiny / CIFAR-100** (rebuttal; LANCE at ε = 0.7,
HOSVD/ASI at the *same per-mode ranks*, mean over 3 seeds). At an identical memory
budget, LANCE matches HOSVD/ASI accuracy while keeping training FLOPs at BP level —
about **100× below iterative HOSVD**:

| Method | 2 layers: MB | GFLOPs | Acc | 4 layers: MB | GFLOPs | Acc |
|--------|----:|----:|----:|----:|----:|----:|
| BP (reference) | 92.34 | 14.87 | 68.77 ± 0.12 | 129.28 | 22.31 | 73.33 ± 0.12 |
| HOSVD | 0.68 | 5.75×10³ | 69.06 ± 0.11 | 1.71 | 6.61×10³ | 70.44 ± 0.10 |
| **LANCE** (ε = 0.7) | 0.68 | 14.53 | **69.23 ± 0.04** | 1.71 | 23.12 | **70.55 ± 0.15** |


## License

This project is released under the MIT License — see [`LICENSE`](LICENSE).
Third-party code retained under its own license is listed in
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES).

## Acknowledgements

This code was developed using the ASI reference implementation,
[Le-TrungNguyen/ICML2025-ASI](https://github.com/Le-TrungNguyen/ICML2025-ASI),
as a reference. In particular, the HOSVD activation-compression layers
(`custom_layers/hosvd_var.py`, `custom_layers/conv_hosvd.py`,
`custom_layers/linear_hosvd.py`) are adapted from that repository, which is
released under the MIT License. The corresponding license notice is retained in
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES) and in the headers of those files.

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{apolinario2025lance,
  title={Lance: Low rank activation compression for efficient on-device continual learning},
  author={Apolinario, Marco Paul E and Roy, Kaushik},
  journal={arXiv preprint arXiv:2509.21617},
  year={2025}
}
```
