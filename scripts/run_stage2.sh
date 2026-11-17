#!/usr/bin/env bash
# Stage 2: zero-shot judge on V, LoRA calibration on the label budget, then
# scoring of the unlabeled pool.
#   run_stage2.sh [H5k|H50k] [model] [prefix] [eval batch size]
# The defaults run the Qwen3-8B judge (run names judge_*). The fallback judge
# of gate G2 is run as: run_stage2.sh H5k Qwen/Qwen3-14B judge14b 16
# (batch 64 runs out of memory with the 14B model on a 48 GB card.)
# Finished steps are skipped, so the script can be restarted.
set -uo pipefail
BUDGET=${1:-H5k}
MODEL=${2:-Qwen/Qwen3-8B}
PREFIX=${3:-judge}
EVAL_BS=${4:-64}
tag=$(echo "$BUDGET" | tr 'A-Z' 'a-z')
quiet() { grep --line-buffered -vE 'Warning|warn\(|Loading weights|Fetching'; }

if [ ! -f "results/${PREFIX}_zeroshot_V/metrics.json" ]; then
  echo "=== ${PREFIX}_zeroshot_V $(date -u +%FT%TZ)"
  python scripts/run_judge.py --name "${PREFIX}_zeroshot_V" --split V --model "$MODEL" --eval-batch-size "$EVAL_BS" 2>&1 | quiet
fi
if ! grep -q '"peak_gpu_gb"' "results/${PREFIX}_$tag/run_metadata.json" 2>/dev/null; then
  echo "=== ${PREFIX}_$tag $(date -u +%FT%TZ)"
  python -m src.train_judge --name "${PREFIX}_$tag" --train "$BUDGET" --model "$MODEL" --eval-batch-size "$EVAL_BS" 2>&1 | quiet
fi
if [ ! -f "checkpoints/${PREFIX}_$tag/adapter_config.json" ]; then
  echo "FAILED ${PREFIX}_$tag: no adapter, stopping"; exit 1
fi
if [ ! -f "results/${PREFIX}_${tag}_U/metrics.json" ]; then
  echo "=== ${PREFIX}_${tag}_U $(date -u +%FT%TZ)"
  python scripts/label_pool.py --name "${PREFIX}_${tag}_U" --adapter "checkpoints/${PREFIX}_$tag" --model "$MODEL" --eval-batch-size "$EVAL_BS" 2>&1 | quiet
fi
echo "STAGE2_QUEUE_DONE $BUDGET $MODEL $(date -u +%FT%TZ)"
