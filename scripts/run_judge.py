"""Score a labeled split with the judge (zero-shot or with a LoRA adapter).

    python scripts/run_judge.py --name judge_zeroshot_V --split V
    python scripts/run_judge.py --name judge_h5k_T --split T --adapter checkpoints/judge_h5k --final

Writes results/<name>/{scores_<split>.parquet, metrics.json, run_metadata.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data import ROOT, load_pairs  # noqa: E402
from src.judge import Judge, judge_metrics, load_judge_config, scores_frame  # noqa: E402
from src.train_student import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="V", choices=["H5k", "H50k", "V", "T"])
    parser.add_argument("--adapter")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, default=0, help="smoke tests only")
    parser.add_argument("--final", action="store_true", help="Stage 5 only: allow the test split")
    args = parser.parse_args()

    cfg = load_judge_config()
    if args.model:
        cfg["model"] = args.model
    frame = load_pairs(args.split, allow_test=args.final)
    if args.limit:
        frame = frame.head(args.limit)
    judge = Judge(cfg, adapter=args.adapter)
    if args.limit:
        print(judge.tokenizer.decode(judge.tokenize(frame["query"][:1], frame["product_text"][:1])[0]))
        print("label token ids:", judge.label_ids)
    probs, timing = judge.predict(frame["query"], frame["product_text"], log_every=50)
    metrics = {**judge_metrics(frame, probs), **timing}

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    scores_frame(frame, probs).to_parquet(out / f"scores_{args.split}.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "run_metadata.json").write_text(json.dumps(
        {"name": args.name, "args": vars(args), "config": cfg, "git_commit": git_commit(),
         "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion"}))
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
