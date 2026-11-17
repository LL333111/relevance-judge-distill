"""Generate and freeze data/splits.json.

Reads only identifier columns: no human label is loaded, for any split.
Refuses to overwrite an existing splits file unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, load_config, load_examples, make_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = ROOT / cfg["splits_path"]
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} exists and is frozen; pass --force to regenerate")

    ids = load_examples(cfg, ["example_id", "query_id", "split"])
    splits = make_splits(ids, cfg)
    payload = {
        "config": {k: cfg[k] for k in ("locale", "seed", "val_queries", "test_queries",
                                        "h5k_pairs", "h50k_pairs")},
        "source": {
            "n_train_pairs": int((ids["split"] == "train").sum()),
            "n_test_pairs": int((ids["split"] == "test").sum()),
            "n_train_queries": int(ids.loc[ids["split"] == "train", "query_id"].nunique()),
            "n_test_queries": int(ids.loc[ids["split"] == "test", "query_id"].nunique()),
        },
        "splits": splits,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")

    print("source:", payload["source"])
    for name, spec in splits.items():
        print(f"{name:5s} queries={spec['n_queries']:6d} pairs={spec['n_pairs']:7d}")


if __name__ == "__main__":
    main()
