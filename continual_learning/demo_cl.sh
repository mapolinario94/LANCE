#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LANCE continual-learning demo: Split CIFAR-100 (AlexNet, 10 tasks).
#
# Trains the 10 tasks sequentially with LANCE's null-space subspace allocation
# and logs per-task accuracy / forgetting to Weights & Biases. CIFAR-100 is
# downloaded/cached under --data-dir on first run.
#
# Hyperparameters match the LANCE Split CIFAR-100 run in the paper (SGD; energy
# thresholds via --threshold-conv / --threshold-linear).
#
# Usage:
#   DATA_DIR=~/Datasets bash demo_cl.sh          # full run
#   EPOCHS=3 bash demo_cl.sh                      # quick smoke run
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-$HOME/Datasets}"
EPOCHS="${EPOCHS:-200}"

echo "=============================================================="
echo ">> Split CIFAR-100  (AlexNet, 10 tasks)"
echo "=============================================================="
python main_splitcifar100.py --model AlexNet --dataset SplitCIFAR100 --n-experiences 10 \
    --data-dir "$DATA_DIR" --dropout --lr 0.01 --batch-size 64 --epochs "$EPOCHS" \
    --threshold-conv 0.98 --threshold-cl 0.97 --print-freq 70 \
    --experiment-name LANCE_split_cifar100

echo "Split CIFAR-100 demo finished."
