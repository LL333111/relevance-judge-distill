import inspect

import numpy as np
import pandas as pd
import pytest

from src import data
from src.data import (EXAMPLES_FILE, ROOT, build_product_text, load_config, load_examples,
                      load_pairs, load_splits, make_splits)

CFG = {"seed": 3, "val_queries": 50, "test_queries": 80, "h5k_pairs": 300, "h50k_pairs": 3000}


def _synthetic_ids(n_train=600, n_test=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    example_id = 0
    for qid in range(n_train + n_test):
        for _ in range(int(rng.integers(8, 41))):
            rows.append((example_id, qid, "train" if qid < n_train else "test"))
            example_id += 1
    return pd.DataFrame(rows, columns=["example_id", "query_id", "split"])


def test_product_text_fields_and_order():
    text = build_product_text("Red  Mug", "Acme", "Red", "Holds 12 oz\n\nDishwasher safe\nB3\nB4")
    assert text == "Title: Red Mug\nBrand: Acme\nColour: Red\n- Holds 12 oz\n- Dishwasher safe\n- B3"


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, "", "nan", "  "])
def test_missing_fields_are_omitted_not_rendered_as_nan(missing):
    text = build_product_text("Red Mug", missing, missing, missing)
    assert text == "Title: Red Mug"
    assert "nan" not in text.lower()


def test_product_text_is_truncated():
    text = build_product_text("x" * 2000, "Acme", None, "bullet")
    assert len(text) == 1200


def test_split_properties_on_synthetic_data():
    ids = _synthetic_ids()
    splits = make_splits(ids, CFG)
    q = {name: set(spec["query_ids"]) for name, spec in splits.items()}
    train_q = set(ids.loc[ids["split"] == "train", "query_id"])
    test_q = set(ids.loc[ids["split"] == "test", "query_id"])

    assert q["H5k"] < q["H50k"]
    for a, b in [("H50k", "V"), ("H50k", "U"), ("V", "U"), ("T", "H50k"), ("T", "V"), ("T", "U")]:
        assert not q[a] & q[b], (a, b)
    assert q["H50k"] | q["V"] | q["U"] == train_q
    assert q["T"] <= test_q

    assert splits["V"]["n_queries"] == 50 and splits["T"]["n_queries"] == 80
    # Budgets are reached with whole queries, overshooting by less than one query.
    assert 300 <= splits["H5k"]["n_pairs"] < 300 + 40
    assert 3000 <= splits["H50k"]["n_pairs"] < 3000 + 40
    total_train = splits["H50k"]["n_pairs"] + splits["V"]["n_pairs"] + splits["U"]["n_pairs"]
    assert total_train == int((ids["split"] == "train").sum())


def test_splits_are_deterministic_and_seed_dependent():
    ids = _synthetic_ids()
    shuffled = ids.sample(frac=1.0, random_state=5)
    assert make_splits(ids, CFG) == make_splits(shuffled, CFG)
    assert make_splits(ids, CFG) != make_splits(ids, {**CFG, "seed": 4})


def test_make_splits_refuses_labels():
    ids = _synthetic_ids().assign(esci_label="E")
    with pytest.raises(ValueError):
        make_splits(ids, CFG)


def test_test_split_is_guarded():
    with pytest.raises(PermissionError):
        load_pairs("T")


def test_unlabeled_pool_never_requests_the_label_column(monkeypatch):
    requested = []

    def fake_load_examples(cfg, columns):
        requested.append(list(columns))
        raise RuntimeError("stop after recording the columns")

    monkeypatch.setattr(data, "load_examples", fake_load_examples)
    monkeypatch.setattr(data, "load_splits", lambda cfg: {"U": {"query_ids": []}})
    with pytest.raises(RuntimeError):
        load_pairs("U", cfg={})
    assert requested and "esci_label" not in requested[0]
    assert "esci_label" not in inspect.signature(make_splits).parameters


# ---- checks on the frozen file (skipped until the data and splits exist)

_cfg = load_config()
_have_frozen = (ROOT / _cfg["splits_path"]).exists() and (ROOT / _cfg["raw_dir"] / EXAMPLES_FILE).exists()
frozen = pytest.mark.skipif(not _have_frozen, reason="raw data or data/splits.json missing")


@frozen
def test_frozen_splits_match_regeneration():
    ids = load_examples(_cfg, ["example_id", "query_id", "split"])
    assert make_splits(ids, _cfg) == load_splits(_cfg)


@frozen
def test_frozen_split_sizes():
    splits = load_splits(_cfg)
    assert splits["V"]["n_queries"] == 1000
    assert splits["T"]["n_queries"] == 3000
    assert 5000 <= splits["H5k"]["n_pairs"] < 5100
    assert 50000 <= splits["H50k"]["n_pairs"] < 50100
    assert 330_000 <= splits["U"]["n_pairs"] <= 370_000
    q = {name: set(spec["query_ids"]) for name, spec in splits.items()}
    assert q["H5k"] < q["H50k"]
    assert not (q["H50k"] & q["V"]) and not (q["H50k"] & q["U"]) and not (q["V"] & q["U"])
    assert not q["T"] & (q["H50k"] | q["V"] | q["U"])
