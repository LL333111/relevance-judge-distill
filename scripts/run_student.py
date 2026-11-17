"""Score a split with the student (zero-shot or with a LoRA adapter).

    python scripts/run_student.py --name row1_zeroshot_V --split V
    python scripts/run_student.py --name row2_V --split V --adapter checkpoints/row2_lr1e-4

Writes results/<name>/{scores_<split>.parquet, metrics.json, run_metadata.json}.
The test split is refused unless --final is given (Stage 5 only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data import ROOT, load_pairs  # noqa: E402
from src.student import Student, load_student_config  # noqa: E402
from src.train_student import evaluate, git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="V", choices=["H5k", "H50k", "V", "T"])
    parser.add_argument("--adapter")
    parser.add_argument("--final", action="store_true", help="Stage 5 only: allow the test split")
    args = parser.parse_args()

    cfg = load_student_config()
    frame = load_pairs(args.split, allow_test=args.final)
    student = Student(cfg, adapter=args.adapter)
    metrics, scores = evaluate(student, frame)

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"example_id": frame["example_id"], "score": scores}).to_parquet(
        out / f"scores_{args.split}.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "run_metadata.json").write_text(json.dumps(
        {"name": args.name, "args": vars(args), "config": cfg, "git_commit": git_commit(),
         "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}, indent=1))
    print(json.dumps(metrics))
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
