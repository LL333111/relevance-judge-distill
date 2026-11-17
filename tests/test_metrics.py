import math

import numpy as np
import pytest

from src.metrics import (GAINS, accuracy, cohen_kappa, confusion_matrix, labels_to_gains,
                         ndcg_at_k, paired_bootstrap_ci)


def test_gains_are_the_esci_task1_values():
    assert GAINS == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def test_gold_as_prediction_gives_one():
    rng = np.random.default_rng(0)
    labels = rng.choice(list("ESCI"), size=2000)
    qids = rng.integers(0, 100, size=2000)
    gains = labels_to_gains(labels)
    result = ndcg_at_k(qids, gains, gains)
    assert result.mean == pytest.approx(1.0, abs=1e-12)
    assert all(v == pytest.approx(1.0, abs=1e-12) for v in result.per_query.values())


def test_hand_computed_value():
    # Ranking by score: S (0.1), E (1.0), I (0). Ideal: E, S, I.
    gains = labels_to_gains(["E", "S", "I"])
    result = ndcg_at_k([7, 7, 7], [0.5, 0.9, 0.1], gains)
    dcg = 0.1 / math.log2(2) + 1.0 / math.log2(3)
    idcg = 1.0 / math.log2(2) + 0.1 / math.log2(3)
    assert result.mean == pytest.approx(dcg / idcg)
    assert result.per_query == {7: pytest.approx(dcg / idcg)}


def test_reversed_gold_is_worse():
    gains = labels_to_gains(list("EESSCII"))
    result = ndcg_at_k([1] * 7, -gains, gains)
    assert result.mean < 0.5


def test_cutoff_ignores_items_below_k():
    # The only relevant item is ranked 11th: nDCG@10 is 0, nDCG@11 is not.
    gains = np.array([0.0] * 10 + [1.0])
    scores = np.arange(11, 0, -1)
    assert ndcg_at_k([1] * 11, scores, gains, k=10).mean == 0.0
    assert ndcg_at_k([1] * 11, scores, gains, k=11).mean > 0.0


def test_queries_without_positive_gain_are_excluded():
    gains = labels_to_gains(["I", "I", "E", "I"])
    result = ndcg_at_k([1, 1, 2, 2], [0.3, 0.2, 0.9, 0.1], gains)
    assert result.n_undefined == 1
    assert result.per_query == {2: pytest.approx(1.0)}
    assert result.mean == pytest.approx(1.0)


def test_ties_are_broken_by_pair_id_and_input_order_is_irrelevant():
    gains = labels_to_gains(["I", "E"])
    # Equal scores: the lower pair id ranks first.
    first = ndcg_at_k([1, 1], [0.5, 0.5], gains, pair_ids=[10, 20])
    second = ndcg_at_k([1, 1], [0.5, 0.5], gains, pair_ids=[20, 10])
    assert first.mean == pytest.approx(1 / math.log2(3))
    assert second.mean == pytest.approx(1.0)
    # Shuffling the rows changes nothing.
    rng = np.random.default_rng(1)
    qids = rng.integers(0, 20, size=400)
    scores = rng.integers(0, 3, size=400).astype(float)
    g = labels_to_gains(rng.choice(list("ESCI"), size=400))
    ids = np.arange(400)
    perm = rng.permutation(400)
    a = ndcg_at_k(qids, scores, g, pair_ids=ids)
    b = ndcg_at_k(qids[perm], scores[perm], g[perm], pair_ids=ids[perm])
    assert a.mean == pytest.approx(b.mean, abs=1e-12)


def test_nan_scores_are_rejected():
    with pytest.raises(ValueError):
        ndcg_at_k([1, 1], [0.1, float("nan")], [1.0, 0.0])


def test_accuracy_and_confusion():
    y_true = list("EESSCI")
    y_pred = list("EESICI")
    assert accuracy(y_true, y_pred) == pytest.approx(5 / 6)
    matrix = confusion_matrix(y_true, y_pred)
    assert matrix.sum() == 6
    assert matrix[1, 3] == 1  # one Substitute predicted as Irrelevant


def test_kappa_known_value():
    # Classic 2x2 example: 20 + 15 agreements out of 50, kappa = 0.4.
    y_true = ["E"] * 25 + ["I"] * 25
    y_pred = ["E"] * 20 + ["I"] * 5 + ["E"] * 10 + ["I"] * 15
    assert cohen_kappa(y_true, y_pred) == pytest.approx(0.4)


def test_kappa_extremes():
    labels = list("ESCI") * 10
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)
    # A constant prediction carries no information.
    assert cohen_kappa(labels, ["E"] * 40) == pytest.approx(0.0)


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    b = {q: float(v) for q, v in enumerate(rng.uniform(0.4, 0.9, size=500))}
    better = {q: v + 0.05 for q, v in b.items()}
    ci = paired_bootstrap_ci(better, b, n_resamples=2000, seed=1)
    assert ci["mean_diff"] == pytest.approx(0.05)
    assert ci["ci_low"] == pytest.approx(0.05) and ci["ci_high"] == pytest.approx(0.05)

    noisy = {q: v + float(rng.normal(0, 0.1)) for q, v in b.items()}
    ci = paired_bootstrap_ci(noisy, b, n_resamples=2000, seed=1)
    assert ci["ci_low"] < 0 < ci["ci_high"]
    assert ci == paired_bootstrap_ci(noisy, b, n_resamples=2000, seed=1)
