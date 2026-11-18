"""Build every reported number from the raw files in results/.

    python scripts/analyze.py            # validation: selection, gates G1 and G2
    python scripts/analyze.py --final    # also the test-set table, CIs and figure

Writes results/summary.json, results/tables.md and (with --final)
results/ndcg_vs_throughput.png. Runs that do not exist are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import ROOT, load_pairs  # noqa: E402
from src.metrics import LABELS, cohen_kappa, confusion_matrix, labels_to_gains, ndcg_at_k, paired_bootstrap_ci  # noqa: E402

RES = ROOT / "results"
JUDGES = {"Qwen3-8B": ("judge_zeroshot_V", "judge_h5k"), "Qwen3-14B": ("judge14b_zeroshot_V", "judge14b_h5k")}
FINAL_JUDGE = "Qwen3-14B"
# Test-set runs (Stage 5): row -> (run name, description).
TEST_ROWS = {
    "1": ("row1_T", "Qwen3-Reranker-0.6B, zero-shot"),
    "2": ("row2_T", "Row 1 + 5K human labels (baseline)"),
    "3": ("row3_T", "Row 1 + 5K human labels + 350K judge labels (method)"),
    "4": ("judge14b_T", "Judge: Qwen3-14B fine-tuned on 5K human labels"),
}


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def scored(run: str, split: str, frame: pd.DataFrame):
    """Frame of the split joined with a run's scores, or None if the run is missing."""
    path = RES / run / f"scores_{split}.parquet"
    if not path.exists():
        return None
    scores = pd.read_parquet(path)
    out = frame.merge(scores, on="example_id", validate="one_to_one")
    assert len(out) == len(frame), f"{run}: scores do not cover split {split}"
    return out


def ndcg(df: pd.DataFrame):
    return ndcg_at_k(df["query_id"], df["score"], labels_to_gains(df["esci_label"]), df["example_id"])


def top10_composition(df: pd.DataFrame) -> dict:
    """Share of the top-10 positions taken by each label, over all queries."""
    order = df.sort_values(["query_id", "score", "example_id"], ascending=[True, False, True])
    top = order.groupby("query_id").head(10)
    return {label: float((top["esci_label"] == label).mean()) for label in LABELS}


def judge_classes(df: pd.DataFrame) -> dict:
    pred = np.array(LABELS)[df[[f"p_{label}" for label in LABELS]].to_numpy().argmax(axis=1)]
    matrix = confusion_matrix(df["esci_label"], pred)
    return {"accuracy": float((pred == df["esci_label"].to_numpy()).mean()),
            "kappa": cohen_kappa(df["esci_label"], pred),
            "recall": {label: float(matrix[i, i] / matrix[i].sum()) for i, label in enumerate(LABELS)},
            "confusion_rows_true_cols_pred": matrix.tolist()}


def student_grid(prefix: str) -> list[dict]:
    rows = []
    for path in sorted(RES.glob(f"{prefix}_lr*/metrics.json")):
        if path.parent.name.endswith("_rerun"):
            continue
        metrics = json.loads(path.read_text())
        best = metrics["best"]
        rows.append({"run": path.parent.name, "lr": path.parent.name.split("_lr")[1],
                     "best_epoch": best["epoch"], "ndcg@10": best["ndcg@10"],
                     "evaluations": len(metrics["history"])})
    return rows


def validation(summary: dict, lines: list[str]) -> None:
    val = load_pairs("V")
    row1 = load_json(RES / "row1_zeroshot_V" / "metrics.json")
    summary["validation"] = {"row1_ndcg@10": row1["ndcg@10"] if row1 else None}

    for row, prefix in (("row2", "row2_h5k"), ("row3", "row3_h5k")):
        grid = student_grid(prefix)
        if grid:
            best = max(grid, key=lambda r: r["ndcg@10"])
            summary["validation"][row] = {"grid": grid, "selected": best["run"]}
            lines += [f"### Validation: {row} learning-rate grid", "", "| Run | Best epoch | nDCG@10 (V) |", "|---|---|---|"]
            lines += [f"| {r['run']}{' **(selected)**' if r is best else ''} | {r['best_epoch']:g} | {r['ndcg@10']:.4f} |" for r in grid]
            lines.append("")

    row2_run = summary["validation"].get("row2", {}).get("selected")
    row2 = ndcg(scored(row2_run, "V", val)) if row2_run else None
    summary["judges"] = {}
    lines += ["### Judges on validation (gates G1 and G2)", "",
              "| Judge | Kappa zero-shot | Kappa fine-tuned | G1 | nDCG@10 | Diff vs row 2 [95% CI] | G2 |", "|---|---|---|---|---|---|---|"]
    for name, (zero_run, tuned_run) in JUDGES.items():
        zero, tuned = scored(zero_run, "V", val), scored(tuned_run, "V", val)
        if zero is None or tuned is None:
            continue
        z, t = judge_classes(zero), judge_classes(tuned)
        result = ndcg(tuned)
        entry = {"zero_shot": {**z, "ndcg@10": ndcg(zero).mean}, "fine_tuned": {**t, "ndcg@10": result.mean},
                 "best_epoch": load_json(RES / tuned_run / "metrics.json")["best"]["epoch"],
                 "G1_pass": t["kappa"] > z["kappa"]}
        ci_text, g2_text = "n/a", "n/a"
        if row2 is not None:
            ci = paired_bootstrap_ci(result.per_query, row2.per_query)
            entry["vs_row2"] = ci
            entry["G2_pass"] = ci["ci_low"] > 0
            ci_text = f"{ci['mean_diff']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]"
            g2_text = "pass" if entry["G2_pass"] else "fail"
        summary["judges"][name] = entry
        lines.append(f"| {name} | {z['kappa']:.3f} | {t['kappa']:.3f} | {'pass' if entry['G1_pass'] else 'fail'} "
                     f"| {result.mean:.4f} | {ci_text} | {g2_text} |")
    lines.append("")

    label_metrics = sorted((RES / "judge14b_h5k_U").glob("metrics*.json")) if (RES / "judge14b_h5k_U").exists() else []
    if label_metrics:
        parts = [json.loads(p.read_text()) for p in label_metrics]
        meta = [json.loads(p.read_text()) for p in sorted((RES / "judge14b_h5k_U").glob("run_metadata*.json"))]
        summary["pool_labeling"] = {"shards": len(parts), "pairs": sum(p["n_pairs"] for p in parts),
                                    "gpu_hours": sum(p["seconds"] for p in parts) / 3600,
                                    "pairs_per_second_per_gpu": [p["pairs_per_second"] for p in parts],
                                    "gpus": sorted({m["gpu"] for m in meta})}


def final(summary: dict, lines: list[str]) -> None:
    test = load_pairs("T", allow_test=True)
    n_queries = test["query_id"].nunique()
    rows, per_query = {}, {}
    for row, (run, description) in TEST_ROWS.items():
        df = scored(run, "T", test)
        if df is None:
            continue
        result = ndcg(df)
        metrics = load_json(RES / run / "metrics.json")
        meta = load_json(RES / run / "run_metadata.json")
        per_query[row] = result.per_query
        rows[row] = {"run": run, "description": description, "ndcg@10": result.mean,
                     "n_queries": len(result.per_query), "n_undefined": result.n_undefined,
                     "pairs_per_second": metrics["pairs_per_second"],
                     "ms_per_query": 1000 * metrics["seconds"] / n_queries, "gpu": meta["gpu"],
                     "top10_composition": top10_composition(df)}
        if "p_E" in df.columns:
            rows[row]["classes"] = judge_classes(df)
    summary["test"] = {"rows": rows, "n_pairs": len(test), "n_queries": n_queries}

    comparisons = {}
    for a, b in (("3", "2"), ("2", "1"), ("3", "1"), ("4", "3"), ("4", "2")):
        if a in per_query and b in per_query:
            comparisons[f"row{a}_minus_row{b}"] = paired_bootstrap_ci(per_query[a], per_query[b])
    summary["test"]["comparisons"] = comparisons

    lines += ["### Main table (test set T, first and only evaluation)", "",
              "| Row | Configuration | nDCG@10 | Pairs/s | ms/query |", "|---|---|---|---|---|"]
    lines += [f"| {row} | {r['description']} | {r['ndcg@10']:.4f} | {r['pairs_per_second']:.0f} | {r['ms_per_query']:.0f} |"
              for row, r in rows.items()]
    lines += ["", "### Paired bootstrap over queries (10,000 resamples)", "", "| Comparison | Difference in nDCG@10 | 95% CI |", "|---|---|---|"]
    lines += [f"| {k.replace('_', ' ')} | {v['mean_diff']:+.4f} | [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] |" for k, v in comparisons.items()]
    lines += ["", "### Share of top-10 positions by human label (test set)", "", "| Row | E | S | C | I |", "|---|---|---|---|---|"]
    lines += [f"| {row} | " + " | ".join(f"{r['top10_composition'][label]:.3f}" for label in LABELS) + " |" for row, r in rows.items()]
    lines.append("")
    if "4" in rows and "classes" in rows["4"]:
        c = rows["4"]["classes"]
        # 20 pairs where the judge's label differs from the human label, for manual review.
        judge = scored(rows["4"]["run"], "T", test)
        judge["judge_label"] = np.array(LABELS)[judge[[f"p_{label}" for label in LABELS]].to_numpy().argmax(axis=1)]
        sample = judge[judge["judge_label"] != judge["esci_label"]].sample(20, random_state=0)
        sample = sample.assign(product_text=sample["product_text"].str.slice(0, 500))
        cols = ["example_id", "query", "product_text", "esci_label", "judge_label"] + [f"p_{label}" for label in LABELS]
        (RES / "judge_disagreements_sample.json").write_text(
            json.dumps(sample[cols].to_dict(orient="records"), indent=1, ensure_ascii=False), encoding="utf-8")
        summary["test"]["judge_disagreement_rate"] = float((judge["judge_label"] != judge["esci_label"]).mean())
        lines += [f"Judge on T: accuracy {c['accuracy']:.3f}, kappa {c['kappa']:.3f}; recall "
                  + ", ".join(f"{label} {c['recall'][label]:.3f}" for label in LABELS) + ".", ""]

    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for row, r in rows.items():
            judge = row == "4"
            ax.scatter(r["pairs_per_second"], r["ndcg@10"], s=70, marker="s" if judge else "o",
                       color="#b3541e" if judge else "#1f5f8b", zorder=3)
            ax.annotate(f"Row {row}", (r["pairs_per_second"], r["ndcg@10"]), textcoords="offset points", xytext=(8, 6))
        ax.set_xscale("log")
        ax.set_xlabel("Throughput (pairs scored per second, log scale)")
        ax.set_ylabel("nDCG@10 on the test set")
        ax.set_title("Quality vs. throughput (square: judge, circles: 0.6B students)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RES / "ndcg_vs_throughput.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="Stage 5: read the test split")
    args = parser.parse_args()
    summary, lines = {}, ["# Result tables (generated by scripts/analyze.py)", ""]
    validation(summary, lines)
    if args.final:
        final(summary, lines)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    (RES / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
