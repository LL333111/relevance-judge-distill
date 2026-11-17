"""LoRA fine-tuning of the student with Margin-MSE.

One loss for every arm:

    L = mean over queries of mean over product pairs (i, j) within the query of
        ((t_i - t_j) - (s_i - s_j))^2

s is the student's P(yes); t is the target on the same 0-1 scale. Targets come
from a `target` column: the gain of the human label (baseline arm) or the
judge's expected gain (method arm). Nothing else differs between arms.

Usage:
    python -m src.train_student --name row2 --train H5k --lr 1e-4 --epochs 8          # baseline
    python -m src.train_student --name row3 --train H5k U --lr 1e-4 --epochs 1 \
        --judge-scores results/judge_h5k_U/scores_U.parquet                           # method
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def margin_mse_loss(scores: torch.Tensor, targets: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
    """Margin-MSE over all pairs within each group (query), averaged over groups.

    scores and targets are 1-D and ordered group by group; group_sizes gives the
    number of candidates in each group. Groups with one candidate have no pair
    and are skipped.
    """
    if scores.shape != targets.shape or scores.dim() != 1:
        raise ValueError("scores and targets must be 1-D and equally long")
    if sum(group_sizes) != len(scores):
        raise ValueError("group_sizes do not add up to the number of scores")
    losses = []
    start = 0
    for n in group_sizes:
        s, t = scores[start:start + n], targets[start:start + n]
        start += n
        if n < 2:
            continue
        error = (t[:, None] - t[None, :]) - (s[:, None] - s[None, :])
        # The diagonal is zero and (i, j), (j, i) contribute equally, so the sum
        # over the full matrix divided by n(n-1) is the mean over pairs i < j.
        losses.append(error.pow(2).sum() / (n * (n - 1)))
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def pack_queries(query_sizes: list[int], order, max_pairs: int) -> list[list[int]]:
    """Pack whole queries (in the given order) into steps of at most max_pairs pairs.

    A query larger than max_pairs gets a step of its own.
    """
    steps, current, count = [], [], 0
    for q in order:
        size = query_sizes[q]
        if current and count + size > max_pairs:
            steps.append(current)
            current, count = [], 0
        current.append(int(q))
        count += size
    if current:
        steps.append(current)
    return steps


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        marker = ROOT / ".git_commit"  # written by scripts/pod.sh push
        return marker.read_text().strip() if marker.exists() else "unknown"


def load_training_frame(names: list[str], judge_scores: str | None = None):
    """Pairs with a `target` column.

    Labeled splits (H5k, H50k) get the gain of their human label. The
    unlabeled pool U gets the judge's expected gain from `judge_scores`
    (a scores_U.parquet written by scripts/label_pool.py).
    """
    import pandas as pd

    from src.data import load_pairs
    from src.metrics import labels_to_gains

    frames = []
    for name in names:
        df = load_pairs(name)
        if name == "U":
            if judge_scores is None:
                raise ValueError("training on U needs --judge-scores")
            scores = pd.read_parquet(ROOT / judge_scores, columns=["example_id", "score"])
            df = df.merge(scores, on="example_id", how="left", validate="one_to_one")
            if df["score"].isna().any():
                raise RuntimeError("judge scores do not cover the pool")
            df["target"] = df["score"]
        else:
            df["target"] = labels_to_gains(df["esci_label"])
        frames.append(df[["example_id", "query_id", "query", "product_text", "target"]])
    return pd.concat(frames, ignore_index=True)


def evaluate(student, frame) -> tuple[dict, np.ndarray]:
    from src.metrics import labels_to_gains, ndcg_at_k

    scores, timing = student.score(frame["query"], frame["product_text"])
    result = ndcg_at_k(frame["query_id"], scores, labels_to_gains(frame["esci_label"]), frame["example_id"])
    return {"ndcg@10": result.mean, "n_queries": len(result.per_query),
            "n_undefined": result.n_undefined, **timing}, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--train", nargs="+", default=["H5k"])
    parser.add_argument("--judge-scores", help="scores_U.parquet; required when training on U")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--evals-per-epoch", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke tests only")
    args = parser.parse_args()

    import pandas as pd

    from src.data import load_pairs
    from src.student import Student, load_student_config

    cfg = load_student_config()
    tcfg = cfg["train"]
    out = ROOT / "results" / args.name
    ckpt = ROOT / "checkpoints" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(tcfg["seed"])
    rng = np.random.default_rng(tcfg["seed"])

    train = load_training_frame(args.train, args.judge_scores)
    val = load_pairs("V")
    student = Student(cfg, train_lora=True)
    params = [p for p in student.peft_model.parameters() if p.requires_grad]

    groups = [g for _, g in train.groupby("query_id", sort=True)]
    sizes = [len(g) for g in groups]
    tokens = [student.tokenize(g["query"], g["product_text"]) for g in groups]
    targets = [torch.tensor(g["target"].to_numpy(), dtype=torch.float32) for g in groups]

    steps_per_epoch = len(pack_queries(sizes, range(len(groups)), tcfg["max_pairs_per_step"]))
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, int(tcfg["warmup_ratio"] * total_steps))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=tcfg["weight_decay"])
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    metadata = {"name": args.name, "args": vars(args), "config": cfg, "git_commit": git_commit(),
                "n_train_pairs": len(train), "n_train_queries": len(groups),
                "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                "n_trainable_params": int(sum(p.numel() for p in params)),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "command": " ".join(sys.argv)}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=1))
    print(json.dumps({k: metadata[k] for k in ("n_train_pairs", "n_train_queries", "steps_per_epoch",
                                               "total_steps", "n_trainable_params", "gpu")}), flush=True)

    history, best = [], {"ndcg@10": -1.0}
    step, running, started = 0, [], time.perf_counter()

    def run_eval():
        nonlocal best
        metrics, scores = evaluate(student, val)
        record = {"step": step, "epoch": step / steps_per_epoch,
                  "train_loss": float(np.mean(running)) if running else None,
                  "train_seconds": time.perf_counter() - started, **metrics}
        running.clear()
        history.append(record)
        if metrics["ndcg@10"] > best["ndcg@10"]:
            best = record
            student.peft_model.save_pretrained(str(ckpt))
            pd.DataFrame({"example_id": val["example_id"], "score": scores}).to_parquet(
                out / "scores_V.parquet", index=False)
        (out / "metrics.json").write_text(json.dumps({"best": best, "history": history}, indent=1))
        print(json.dumps(record), flush=True)
        student.peft_model.train()

    student.peft_model.train()
    done = False
    while not done:
        for step_queries in pack_queries(sizes, rng.permutation(len(groups)), tcfg["max_pairs_per_step"]):
            batch = student.collate([ids for q in step_queries for ids in tokens[q]])
            target = torch.cat([targets[q] for q in step_queries]).to(student.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = student.prob_yes(batch)
            loss = margin_mse_loss(scores, target, [sizes[q] for q in step_queries])
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            running.append(loss.item())
            step += 1
            if step % eval_every == 0 or step == total_steps:
                run_eval()
            if step >= total_steps:
                done = True
                break

    metadata["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    metadata["lora_b_norm"] = float(sum(p.detach().float().pow(2).sum() for n, p in
                                        student.peft_model.named_parameters() if "lora_B" in n) ** 0.5)
    metadata["max_query_size"] = int(max(sizes))
    metadata["train_seconds"] = time.perf_counter() - started
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=1))
    print("BEST", json.dumps(best), flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
