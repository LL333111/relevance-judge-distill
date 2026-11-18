#!/usr/bin/env bash
# Row 3 (method): fine-tune the student on H5k (human labels) plus the pool U
# with the judge's expected gain as the target.
#   run_stage3.sh <scores_U.parquet> [learning rates...]
# Same four learning rates as row 2 and the same loss. One epoch over the
# ~355K pairs, validated on V four times. Passing a subset of the learning
# rates lets several GPUs share the four runs.
set -uo pipefail
SCORES=$1; shift
[ $# -gt 0 ] || set -- 5e-5 1e-4 2e-4 4e-4
for lr in "$@"; do
  name=row3_h5k_lr$lr
  if grep -q '"peak_gpu_gb"' "results/$name/run_metadata.json" 2>/dev/null; then
    echo "skip $name (finished)"; continue
  fi
  echo "=== $name $(date -u +%FT%TZ)"
  python -m src.train_student --name "$name" --train H5k U --judge-scores "$SCORES" --lr "$lr" \
    --epochs 1 --evals-per-epoch 4 2>&1 | grep --line-buffered -vE 'Warning|warn\(|Loading weights' || echo "FAILED $name"
done
echo "STAGE3_QUEUE_DONE $* $(date -u +%FT%TZ)"
