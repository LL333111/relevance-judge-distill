"""Score the unlabeled pool U with the calibrated judge.

    python scripts/label_pool.py --name judge_h5k_U --adapter checkpoints/judge_h5k

U is loaded without its human-label column. Writes
results/<name>/scores_U.parquet (four class probabilities and the expected
gain per pair) and the labeling throughput.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data import ROOT, load_pairs  # noqa: E402
from src.judge import Judge, load_judge_config, scores_frame  # noqa: E402
from src.train_student import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model")
    parser.add_argument("--eval-batch-size", type=int,
                        help="inference batch size; lower it when the model is close to GPU memory")
    args = parser.parse_args()

    cfg = load_judge_config()
    if args.model:
        cfg["model"] = args.model
    if args.eval_batch_size:
        cfg["eval_batch_size"] = args.eval_batch_size
    pool = load_pairs("U")
    assert "esci_label" not in pool.columns
    judge = Judge(cfg, adapter=args.adapter)
    probs, timing = judge.predict(pool["query"], pool["product_text"], log_every=200)

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    scores_frame(pool, probs).to_parquet(out / "scores_U.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(timing, indent=1))
    (out / "run_metadata.json").write_text(json.dumps(
        {"name": args.name, "args": vars(args), "config": cfg, "git_commit": git_commit(),
         "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}, indent=1))
    print(json.dumps(timing))
    print("LABEL_DONE")


if __name__ == "__main__":
    main()
