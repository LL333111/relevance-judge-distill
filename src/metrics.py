"""Ranking and agreement metrics.

nDCG@10 uses the ESCI Task 1 gains directly as gains (no 2^rel - 1), with the
usual log2(rank + 1) discount. Within a query, candidates are sorted by score
descending; ties are broken by pair identifier ascending, so results are
deterministic.

Queries whose candidates all have gain 0 have an ideal DCG of 0, so nDCG is
undefined for them. They are excluded from the mean and counted separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LABELS = ("E", "S", "C", "I")
GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def labels_to_gains(labels) -> np.ndarray:
    return np.array([GAINS[label] for label in labels], dtype=np.float64)


@dataclass
class NdcgResult:
    mean: float
    per_query: dict  # query_id -> nDCG@k, defined queries only
    n_undefined: int  # queries with no candidate of positive gain


def ndcg_at_k(query_ids, scores, gains, pair_ids=None, k: int = 10) -> NdcgResult:
    """Mean nDCG@k over queries.

    query_ids, scores, gains and pair_ids are parallel arrays, one entry per
    query-product pair. pair_ids breaks score ties (ascending); if omitted,
    the input order is used.
    """
    query_ids = np.asarray(query_ids)
    scores = np.asarray(scores, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    n = len(query_ids)
    if not (len(scores) == len(gains) == n):
        raise ValueError("query_ids, scores and gains must have the same length")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    pair_ids = np.arange(n) if pair_ids is None else np.asarray(pair_ids)

    # Sort by query, then score descending, then pair id ascending.
    order = np.lexsort((pair_ids, -scores, query_ids))
    q_sorted = query_ids[order]
    g_sorted = gains[order]
    starts = np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1]])
    ends = np.r_[starts[1:], n]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    per_query = {}
    n_undefined = 0
    for start, end in zip(starts, ends):
        g = g_sorted[start:end]
        ideal = np.sort(g)[::-1][:k]
        idcg = float(ideal @ discounts[: len(ideal)])
        if idcg <= 0.0:
            n_undefined += 1
            continue
        top = g[:k]
        dcg = float(top @ discounts[: len(top)])
        per_query[q_sorted[start].item()] = dcg / idcg

    mean = float(np.mean(list(per_query.values()))) if per_query else float("nan")
    return NdcgResult(mean=mean, per_query=per_query, n_undefined=n_undefined)


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and equally long")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=LABELS) -> np.ndarray:
    """Rows are true labels, columns are predicted labels, in `labels` order."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    return matrix


def cohen_kappa(y_true, y_pred, labels=LABELS) -> float:
    """Unweighted Cohen's kappa."""
    matrix = confusion_matrix(y_true, y_pred, labels).astype(np.float64)
    total = matrix.sum()
    if total == 0:
        raise ValueError("no observations")
    observed = np.trace(matrix) / total
    expected = float(matrix.sum(axis=1) @ matrix.sum(axis=0)) / total**2
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))


def paired_bootstrap_ci(per_query_a: dict, per_query_b: dict, n_resamples: int = 10_000,
                        seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI of mean(a - b), resampling queries with replacement.

    Only queries present in both inputs are used (they should be identical).
    """
    common = sorted(set(per_query_a) & set(per_query_b))
    if not common:
        raise ValueError("no common queries")
    diff = np.array([per_query_a[q] - per_query_b[q] for q in common])
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 500):
        stop = min(start + 500, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_queries": n, "n_resamples": n_resamples}
