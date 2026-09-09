# LANCE — Continual Learning

Continual-learning (CL) experiments for **LANCE**, corresponding to the
continual-learning results in the paper (Split CIFAR-100, Split MiniImageNet,
5-Datasets).

In the CL setting, tasks arrive sequentially. After each task LANCE keeps, per
layer, a compact set of **important directions** on the last activation mode.
When calibrating the fixed HOSVD subspace for the next task, the new factors are
constrained to the **null space** of those stored directions, so a new task
learns in directions orthogonal to previous tasks — mitigating catastrophic
forgetting **without** keeping a large task-memory matrix in SRAM during
training (the memory advantage over gradient-projection methods like GPM/CODE-CL).

> This is the standalone CL codebase. For the single-task fine-tuning experiments
> (MCUNet / MobileNetV2 / ResNet / DeiT), see the parent directory.

## Benchmarks & models

| Entry point               | Benchmark          | Model    | Tasks |
|---------------------------|--------------------|----------|-------|
| `main_splitcifar100.py`   | Split CIFAR-100    | AlexNet  | 10    |
| `main.py`                 | Split MiniImageNet | ResNet18 | 20    |
| `main_5datasets.py`       | 5-Datasets         | ResNet18 | 5     |


## Installation

```bash
pip install -r requirements.txt
```

This part depends on **[avalanche-lib](https://avalanche.continualai.org/)**
(`0.5.0`) for the CL benchmarks/datasets/metrics — a heavier dependency than the
fine-tuning experiments. A GPU is required (`.cuda()` is used throughout).

## Data

- **Split CIFAR-100** and **5-Datasets** (CIFAR-10, MNIST, SVHN, Fashion-MNIST,
  notMNIST) are downloaded/cached under `--data-dir` (default `~/Datasets`) on
  first run.
- **Split MiniImageNet** is built by avalanche from a local **ImageNet-2012**
  directory. Point to it with `--imagenet-path /path/to/imagenet2012`.

## Experiment logging (Weights & Biases)

Runs log per-task accuracy and forgetting to **wandb**; `--experiment-name` is
used as the wandb project. Use `wandb login`, or `export WANDB_MODE=offline`.

## Running

`demo_cl.sh` **demonstrates** LANCE on Split CIFAR-100:

```bash
DATA_DIR=~/Datasets bash demo_cl.sh          # full run
EPOCHS=3 bash demo_cl.sh                      # quick smoke run
```

The other two benchmarks run analogously through their own entry points —
`main.py` for Split MiniImageNet (needs `--imagenet-path`) and `main_5datasets.py`
for 5-Datasets. Each uses SGD with per-layer LANCE energy thresholds
`--threshold-conv` / `--threshold-linear` (ε); `--threshold-inc` adds a small
per-task increment. See each entry point's `--help` for the full option list.

## Acknowledgements

This continual-learning code is **adapted from the CODE-CL repository**
([mapolinario94/CODE-CL](https://github.com/mapolinario94/CODE-CL), ICCV 2025,
[arXiv:2411.15235](https://arxiv.org/abs/2411.15235)), by the same author and
released under the MIT License. We reuse its training/evaluation harness and
benchmark plumbing, and replace the conceptor-based gradient projection with
LANCE's fixed low-rank, null-space activation subspaces. See [`LICENSE`](LICENSE).

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
