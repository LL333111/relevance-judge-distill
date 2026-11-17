"""ESCI loading, product text and query-level splits.

Access rules enforced here:
  * U (the unlabeled pool) is loaded without ever reading the esci_label
    column from disk.
  * T (test) can only be loaded with allow_test=True, which is reserved for
    the Stage 5 final evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"
SPLIT_NAMES = ("H5k", "H50k", "V", "U", "T")
EMPTY_PRODUCT_TEXT = "Title: (no product information available)"
_ID_COLUMNS = ["example_id", "query_id", "product_id", "product_locale", "small_version", "split"]


def load_config(path: str | Path = "configs/data.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- product text

def _clean(value) -> str:
    """Stripped text, or '' for missing values (None, NaN, the string 'nan')."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if value is pd.NA:
        return ""
    text = " ".join(str(value).split())
    return "" if text.lower() in ("nan", "none") else text


def build_product_text(title, brand=None, color=None, bullet_point=None,
                       max_bullets: int = 3, max_chars: int = 1200) -> str:
    """Title, brand, colour and the first bullet points, truncated.

    Missing fields are omitted entirely; they are never rendered as "nan".
    """
    parts = []
    title = _clean(title)
    if title:
        parts.append(f"Title: {title}")
    brand = _clean(brand)
    if brand:
        parts.append(f"Brand: {brand}")
    color = _clean(color)
    if color:
        parts.append(f"Colour: {color}")
    if bullet_point is not None and _clean(bullet_point):
        bullets = [_clean(line) for line in str(bullet_point).split("\n")]
        bullets = [b for b in bullets if b][:max_bullets]
        parts.extend(f"- {b}" for b in bullets)
    return "\n".join(parts)[:max_chars].rstrip()


# ------------------------------------------------------------------- examples

def load_examples(cfg: dict, columns: list[str]) -> pd.DataFrame:
    """US Task 1 rows with the requested columns, duplicates of example_id removed."""
    needed = sorted(set(columns) | {"example_id", "product_locale", "small_version"})
    df = pd.read_parquet(
        ROOT / cfg["raw_dir"] / EXAMPLES_FILE,
        columns=needed,
        filters=[("product_locale", "==", cfg["locale"]), ("small_version", "==", 1)],
    )
    df = df.drop_duplicates("example_id").sort_values("example_id").reset_index(drop=True)
    return df[columns]


def example_ids_hash(example_ids) -> str:
    """Order-independent fingerprint of a set of example ids."""
    ids = np.sort(np.asarray(example_ids, dtype=np.int64))
    return hashlib.sha256(ids.tobytes()).hexdigest()


# --------------------------------------------------------------------- splits

def make_splits(ids: pd.DataFrame, cfg: dict) -> dict:
    """Assign whole queries to H5k, H50k, V, U (from train) and T (from test).

    `ids` needs the columns example_id, query_id and split; labels are not
    needed and must not be passed. Returns {name: {"query_ids", "n_queries",
    "n_pairs", "example_ids_sha256"}}. H5k is a subset of H50k; every other
    pair of splits is query-disjoint.
    """
    if "esci_label" in ids.columns:
        raise ValueError("make_splits must not see labels")
    sizes = ids.groupby("query_id").size()
    seed = int(cfg["seed"])

    train_q = np.sort(ids.loc[ids["split"] == "train", "query_id"].unique())
    test_q = np.sort(ids.loc[ids["split"] == "test", "query_id"].unique())
    if set(train_q) & set(test_q):
        raise ValueError("official train and test share queries")

    train_q = np.random.default_rng(seed).permutation(train_q)
    n_val = int(cfg["val_queries"])
    val_q, rest = train_q[:n_val], train_q[n_val:]

    cum_pairs = np.cumsum(sizes.loc[rest].to_numpy())
    # Smallest number of whole queries whose pairs reach each budget.
    n_h5k = int(np.searchsorted(cum_pairs, int(cfg["h5k_pairs"])) + 1)
    n_h50k = int(np.searchsorted(cum_pairs, int(cfg["h50k_pairs"])) + 1)
    if n_h50k >= len(rest):
        raise ValueError("label budget exhausts the training queries")

    n_test = int(cfg["test_queries"])
    if n_test > len(test_q):
        raise ValueError("not enough test queries")
    test_sample = np.random.default_rng(seed + 1).choice(test_q, size=n_test, replace=False)

    assignment = {
        "H5k": rest[:n_h5k],
        "H50k": rest[:n_h50k],
        "V": val_q,
        "U": rest[n_h50k:],
        "T": test_sample,
    }
    out = {}
    for name, queries in assignment.items():
        queries = np.sort(queries)
        rows = ids[ids["query_id"].isin(queries)]
        out[name] = {
            "n_queries": int(len(queries)),
            "n_pairs": int(len(rows)),
            "example_ids_sha256": example_ids_hash(rows["example_id"]),
            "query_ids": [int(q) for q in queries],
        }
    return out


def load_splits(cfg: dict) -> dict:
    with open(ROOT / cfg["splits_path"], encoding="utf-8") as f:
        return json.load(f)["splits"]


# ---------------------------------------------------------------------- pairs

def _product_texts(cfg: dict) -> pd.DataFrame:
    """product_id -> product_text for the configured locale (cached)."""
    cache = ROOT / cfg["cache_dir"] / f"product_text_{cfg['locale']}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    products = pd.read_parquet(
        ROOT / cfg["raw_dir"] / PRODUCTS_FILE,
        columns=["product_id", "product_title", "product_bullet_point",
                 "product_brand", "product_color", "product_locale"],
        filters=[("product_locale", "==", cfg["locale"])],
    ).drop_duplicates("product_id")
    text_cfg = cfg["product_text"]
    products["product_text"] = [
        build_product_text(t, b, c, bp, text_cfg["max_bullets"], text_cfg["max_chars"])
        for t, b, c, bp in zip(products["product_title"], products["product_brand"],
                               products["product_color"], products["product_bullet_point"])
    ]
    out = products[["product_id", "product_text"]].reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def load_pairs(name: str, cfg: dict | None = None, allow_test: bool = False) -> pd.DataFrame:
    """Query-product pairs of one split, with product text.

    Columns: example_id, query_id, query, product_id, product_text, and
    esci_label for every split except U. The label column of U is never read.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}")
    if name == "T" and not allow_test:
        raise PermissionError("T is reserved for the Stage 5 final evaluation (allow_test=True)")
    cfg = cfg or load_config()
    spec = load_splits(cfg)[name]

    columns = ["example_id", "query_id", "query", "product_id", "split"]
    if name != "U":
        columns.append("esci_label")
    df = load_examples(cfg, columns)
    official = "test" if name == "T" else "train"
    df = df[(df["split"] == official) & df["query_id"].isin(spec["query_ids"])]
    if example_ids_hash(df["example_id"]) != spec["example_ids_sha256"]:
        raise RuntimeError(f"split {name} does not match the frozen data/splits.json")

    df = df.merge(_product_texts(cfg), on="product_id", how="left", validate="many_to_one")
    if df["product_text"].isna().any():
        raise RuntimeError(f"split {name} has pairs whose product is not in the catalogue")
    # A handful of catalogue rows have no title, brand, colour or bullets. The
    # pair is kept (dropping it would change the candidate sets) with a
    # placeholder text.
    df["product_text"] = df["product_text"].where(df["product_text"] != "", EMPTY_PRODUCT_TEXT)
    return df.drop(columns="split").sort_values("example_id").reset_index(drop=True)
