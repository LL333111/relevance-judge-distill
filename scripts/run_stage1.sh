#!/usr/bin/env bash
# Row 2 (baseline): fine-tune the student on human labels.
#   run_stage1.sh [H5k|H50k] [epochs] [evals per epoch]
# Four configurations, fixed before any of
# them ran: four learning rates, validated on V. The main table uses
# H5k with 8 epochs; the ablation uses H50k with 2 epochs and two evaluations
# per epoch (the H5k runs all peaked within their first 176 steps).
# Finished runs are skipped, so the script can be restarted.
set -uo pipefail
BUDGET=${1:-H5k}
EPOCHS=${2:-8}
EVALS=${3:-1}
tag=$(echo "$BUDGET" | tr 'A-Z' 'a-z')
for lr in 5e-5 1e-4 2e-4 4e-4; do
  name=row2_${tag}_lr$lr
  if grep -q '"peak_gpu_gb"' "results/$name/run_metadata.json" 2>/dev/null; then
    echo "skip $name (finished)"; continue
  fi
  echo "=== $name $(date -u +%FT%TZ)"
  python -m src.train_student --name "$name" --train "$BUDGET" --lr "$lr" --epochs "$EPOCHS" \
    --evals-per-epoch "$EVALS" 2>&1 | grep --line-buffered -vE 'Warning|warn\(|Loading weights' || echo "FAILED $name"
done
echo "STAGE1_QUEUE_DONE $BUDGET $(date -u +%FT%TZ)"
