"""LoRA fine-tuning of the judge on human labels.

The loss is cross-entropy over the four label tokens at the final position,
the same four-way distribution that the readout uses.

    python -m src.train_judge --name judge_h5k --train H5k
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import torch

from src.judge import Judge, judge_metrics, load_judge_config, scores_frame
from src.metrics import LABELS
from src.train_student import ROOT, git_commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--train", default="H5k", choices=["H5k", "H50k"])
    parser.add_argument("--model")
    parser.add_argument("--eval-batch-size", type=int,
                        help="inference batch size; lower it when the model is close to GPU memory")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke tests only")
    args = parser.parse_args()

    from src.data import load_pairs

    cfg = load_judge_config()
    if args.model:
        cfg["model"] = args.model
    if args.eval_batch_size:
        cfg["eval_batch_size"] = args.eval_batch_size
    tcfg = cfg["train"]
    epochs = args.epochs or tcfg["epochs"]
    out = ROOT / "results" / args.name
    ckpt = ROOT / "checkpoints" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(tcfg["seed"])
    rng = np.random.default_rng(tcfg["seed"])

    train = load_pairs(args.train)
    val = load_pairs("V")
    judge = Judge(cfg, train_lora=True)
    params = [p for p in judge.peft_model.parameters() if p.requires_grad]
    tokens = judge.tokenize(train["query"], train["product_text"])
    targets = torch.tensor([LABELS.index(label) for label in train["esci_label"]])

    batch_size = tcfg["batch_size"]
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps = args.max_steps or steps_per_epoch * epochs
    warmup = max(1, int(tcfg["warmup_ratio"] * total_steps))
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    metadata = {"name": args.name, "args": vars(args), "config": cfg, "git_commit": git_commit(),
                "n_train_pairs": len(train), "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "command": " ".join(sys.argv)}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=1))
    print(json.dumps({k: metadata[k] for k in ("n_train_pairs", "steps_per_epoch", "total_steps", "gpu")}),
          flush=True)

    history, best = [], {"ndcg@10": -1.0}
    step, running, started = 0, [], time.perf_counter()

    def run_eval():
        nonlocal best
        probs, timing = judge.predict(val["query"], val["product_text"])
        record = {"step": step, "epoch": step / steps_per_epoch, "train_loss": float(np.mean(running)),
                  "train_seconds": time.perf_counter() - started, **judge_metrics(val, probs), **timing}
        running.clear()
        history.append(record)
        if record["ndcg@10"] > best["ndcg@10"]:
            best = record
            judge.peft_model.save_pretrained(str(ckpt))
            scores_frame(val, probs).to_parquet(out / "scores_V.parquet", index=False)
        (out / "metrics.json").write_text(json.dumps({"best": best, "history": history}, indent=1))
        print(json.dumps({k: v for k, v in record.items() if k != "confusion"}), flush=True)
        judge.peft_model.train()

    judge.peft_model.train()
    done = False
    while not done:
        order = rng.permutation(len(train))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = judge.collate([tokens[j] for j in idx])
            loss = torch.nn.functional.cross_entropy(judge.label_logits(batch), targets[idx].to(judge.device))
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            running.append(loss.item())
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{total_steps} loss {np.mean(running[-50:]):.4f} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
            if step % steps_per_epoch == 0 or step == total_steps:
                run_eval()
            if step >= total_steps:
                done = True
                break

    metadata["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    metadata["train_seconds"] = time.perf_counter() - started
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=1))
    print("BEST", json.dumps({k: v for k, v in best.items() if k != "confusion"}), flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
