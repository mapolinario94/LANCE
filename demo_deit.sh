#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LANCE demo: fine-tuning DeiT-Tiny (ViT) on CIFAR-100.
#
# Illustrates the matched-memory comparison discussed in the paper's rebuttal
# (see the "DeiT-Tiny (ViT)" section of the README): the last N Linear layers
# are fine-tuned with (a) full back-prop, (b) LANCE, and (c) HOSVD run at the
# SAME per-mode ranks as LANCE (iso-memory).
#
# Workflow: LANCE runs first and writes its per-layer ranks to JSON; HOSVD then
# loads those ranks so both methods use an identical activation-memory budget.
#
# Setup (matches the rebuttal script): ImageNet-pretrained deit_tiny_patch16_224
# (timm), SGD, lr 0.05, batch 128, 50 epochs, energy threshold epsilon = 0.7,
# 100 calibration batches, last {2, 4} Linear layers.
# The paper averages over 3 seeds (42, 123, 456); this demo runs a single seed
# by default (override with SEED=...).
#
# Usage:
#   DATA_DIR=~/Datasets bash demo_deit.sh
#   EPOCHS=3 bash demo_deit.sh          # quick smoke run
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- config (override via environment) ----
DATA_DIR="${DATA_DIR:-$HOME/Datasets}"
DATASET="${DATASET:-cifar100}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.05}"
THRESHOLD="${THRESHOLD:-0.7}"      # LANCE energy threshold (epsilon)
WARMUP="${WARMUP:-100}"            # one-shot calibration batches (N)
SEED="${SEED:-42}"

N_UNFROZEN=(2 4)

COMMON="--arch deit_tiny --dataset ${DATASET} --data-dir ${DATA_DIR} \
--optim sgd --lr ${LR} --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --seed ${SEED}"
EXP="deit_tiny_${DATASET}_demo"

for n in "${N_UNFROZEN[@]}"; do
  ranks="ranks_deit_${DATASET}_n${n}.json"

  echo "=============================================================="
  echo ">> DeiT-Tiny / ${DATASET} / last ${n} linears / BP baseline"
  echo "=============================================================="
  python main.py ${COMMON} --n-unfrozen "${n}" --experiment-name "${EXP}"

  echo "=============================================================="
  echo ">> DeiT-Tiny / ${DATASET} / last ${n} linears / LANCE (eps=${THRESHOLD})"
  echo "   (saves per-layer ranks to ${ranks} for the iso-memory HOSVD run)"
  echo "=============================================================="
  python main.py ${COMMON} --n-unfrozen "${n}" \
    --use-lance --lance-threshold "${THRESHOLD}" --lance-warmup-batches "${WARMUP}" \
    --save-lance-ranks "${ranks}" --experiment-name "${EXP}"

  echo "=============================================================="
  echo ">> DeiT-Tiny / ${DATASET} / last ${n} linears / HOSVD (iso-memory)"
  echo "=============================================================="
  python main.py ${COMMON} --n-unfrozen "${n}" \
    --use-hosvd --load-lance-ranks "${ranks}" --experiment-name "${EXP}"
done

echo "All DeiT demo runs finished."
