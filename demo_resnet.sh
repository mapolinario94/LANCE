#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LANCE demo: fine-tuning ResNet18 / ResNet34 on CIFAR-10 / CIFAR-100.
#
# Demonstrates LANCE single-task fine-tuning (this is an illustration, not an
# exact reproduction — results vary with seed and hardware): for each backbone /
# dataset it fine-tunes the last N conv layers with (a) full back-prop (BP
# baseline) and (b) LANCE, and logs accuracy / activation-memory / FLOPs to wandb.
#
# Settings follow the paper: ImageNet-pretrained backbone, SGD, lr 0.05, batch 128,
# 50 epochs, N = 100 one-shot HOSVD calibration batches, energy threshold
# epsilon = 0.7, last {2, 4} conv layers fine-tuned.
#
# Usage:
#   DATA_DIR=~/Datasets bash demo_resnet.sh          # run everything (16 runs)
#   EPOCHS=5 bash demo_resnet.sh                      # quick smoke run
# CIFAR-10/100 are downloaded automatically on first use.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- config (override via environment) ----
DATA_DIR="${DATA_DIR:-$HOME/Datasets}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.05}"
THRESHOLD="${THRESHOLD:-0.7}"      # LANCE energy threshold (epsilon)
WARMUP="${WARMUP:-100}"            # one-shot calibration batches (N)
SEED="${SEED:-42}"

ARCHS=(resnet18 resnet34)
DATASETS=(cifar10 cifar100)
N_UNFROZEN=(2 4)

COMMON="--data-dir ${DATA_DIR} --optim sgd --lr ${LR} --batch-size ${BATCH_SIZE} \
--epochs ${EPOCHS} --seed ${SEED}"

for arch in "${ARCHS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    exp="${arch}_${ds}_demo"
    for n in "${N_UNFROZEN[@]}"; do
      echo "=============================================================="
      echo ">> ${arch} / ${ds} / last ${n} layers / BP baseline"
      echo "=============================================================="
      python main.py --arch "${arch}" --dataset "${ds}" ${COMMON} \
        --n-unfrozen "${n}" --experiment-name "${exp}"

      echo "=============================================================="
      echo ">> ${arch} / ${ds} / last ${n} layers / LANCE (eps=${THRESHOLD})"
      echo "=============================================================="
      python main.py --arch "${arch}" --dataset "${ds}" ${COMMON} \
        --n-unfrozen "${n}" --use-lance --lance-threshold "${THRESHOLD}" \
        --lance-warmup-batches "${WARMUP}" --experiment-name "${exp}"
    done
  done
done

echo "All demo runs finished."
