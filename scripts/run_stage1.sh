#!/usr/bin/env bash
# Row 2 (baseline): fine-tune on H5k with human-label targets.
# Four configurations, fixed before any of
# them ran: four learning rates, 8 epochs each, validated on V every epoch.
# Finished runs are skipped, so the script can be restarted.
set -uo pipefail
EPOCHS=8
for lr in 5e-5 1e-4 2e-4 4e-4; do
  name=row2_h5k_lr$lr
  if grep -q '"peak_gpu_gb"' "results/$name/run_metadata.json" 2>/dev/null; then
    echo "skip $name (finished)"; continue
  fi
  echo "=== $name $(date -u +%FT%TZ)"
  python -m src.train_student --name "$name" --train H5k --lr "$lr" --epochs "$EPOCHS" \
    2>&1 | grep -vE 'Warning|warn\(|Loading weights' || echo "FAILED $name"
done
echo STAGE1_QUEUE_DONE
