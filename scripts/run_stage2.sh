#!/usr/bin/env bash
# Stage 2: zero-shot judge on V, LoRA calibration on the label budget, then
# scoring of the unlabeled pool. Usage: run_stage2.sh [H5k|H50k]
# Finished steps are skipped, so the script can be restarted.
set -uo pipefail
BUDGET=${1:-H5k}
tag=$(echo "$BUDGET" | tr 'A-Z' 'a-z')
quiet() { grep --line-buffered -vE 'Warning|warn\(|Loading weights|Fetching'; }

if [ ! -f results/judge_zeroshot_V/metrics.json ]; then
  echo "=== judge_zeroshot_V $(date -u +%FT%TZ)"
  python scripts/run_judge.py --name judge_zeroshot_V --split V 2>&1 | quiet
fi
if ! grep -q '"peak_gpu_gb"' "results/judge_$tag/run_metadata.json" 2>/dev/null; then
  echo "=== judge_$tag $(date -u +%FT%TZ)"
  python -m src.train_judge --name "judge_$tag" --train "$BUDGET" 2>&1 | quiet
fi
if [ ! -f "results/judge_${tag}_U/metrics.json" ]; then
  echo "=== judge_${tag}_U $(date -u +%FT%TZ)"
  python scripts/label_pool.py --name "judge_${tag}_U" --adapter "checkpoints/judge_$tag" 2>&1 | quiet
fi
echo "STAGE2_QUEUE_DONE $BUDGET $(date -u +%FT%TZ)"
