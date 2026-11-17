import itertools

import pytest

torch = pytest.importorskip("torch")

from src.train_student import margin_mse_loss, pack_queries  # noqa: E402


def brute_force(scores, targets, sizes):
    losses, start = [], 0
    for n in sizes:
        s, t = scores[start:start + n], targets[start:start + n]
        start += n
        pairs = list(itertools.combinations(range(n), 2))
        if pairs:
            losses.append(sum(((t[i] - t[j]) - (s[i] - s[j])) ** 2 for i, j in pairs) / len(pairs))
    return sum(losses) / len(losses)


def test_matches_brute_force_over_pairs():
    torch.manual_seed(0)
    sizes = [5, 1, 8, 2]
    scores, targets = torch.rand(16), torch.rand(16)
    expected = brute_force(scores.tolist(), targets.tolist(), sizes)
    assert margin_mse_loss(scores, targets, sizes).item() == pytest.approx(expected, rel=1e-5)


def test_hand_computed():
    # One query, two products: target margin 0.9, student margin 0.5.
    loss = margin_mse_loss(torch.tensor([0.7, 0.2]), torch.tensor([1.0, 0.1]), [2])
    assert loss.item() == pytest.approx((0.9 - 0.5) ** 2)


def test_zero_when_margins_match_and_shift_invariant():
    targets = torch.tensor([1.0, 0.1, 0.01, 0.0, 1.0])
    assert margin_mse_loss(targets.clone(), targets, [5]).item() == pytest.approx(0.0, abs=1e-12)
    # Only differences matter: a constant offset per query costs nothing.
    assert margin_mse_loss(targets - 0.3, targets, [5]).item() == pytest.approx(0.0, abs=1e-12)


def test_pairs_never_cross_queries():
    scores = torch.tensor([0.9, 0.1, 0.5, 0.5])
    targets = torch.tensor([0.9, 0.1, 0.0, 0.0])
    # Within each query the margins match exactly, although they differ across queries.
    assert margin_mse_loss(scores, targets, [2, 2]).item() == pytest.approx(0.0, abs=1e-12)
    assert margin_mse_loss(scores, targets, [4]).item() > 0.01


def test_each_query_weighs_the_same():
    small = margin_mse_loss(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]), [2])
    scores = torch.cat([torch.zeros(2), torch.rand(30)])
    targets = torch.cat([torch.tensor([1.0, 0.0]), torch.zeros(30)])
    targets[2:] = scores[2:]  # the big query is fitted perfectly
    both = margin_mse_loss(scores, targets, [2, 30])
    assert both.item() == pytest.approx(small.item() / 2)


def test_gradient_moves_scores_toward_target_order():
    scores = torch.tensor([0.5, 0.5], requires_grad=True)
    margin_mse_loss(scores, torch.tensor([1.0, 0.0]), [2]).backward()
    assert scores.grad[0] < 0 < scores.grad[1]  # descent raises the first, lowers the second


def test_single_candidate_queries_give_zero_loss_with_grad():
    scores = torch.tensor([0.3], requires_grad=True)
    loss = margin_mse_loss(scores, torch.tensor([1.0]), [1])
    loss.backward()
    assert loss.item() == 0.0


def test_bad_inputs():
    with pytest.raises(ValueError):
        margin_mse_loss(torch.rand(3), torch.rand(3), [2])
    with pytest.raises(ValueError):
        margin_mse_loss(torch.rand(3), torch.rand(4), [3])


def test_pack_queries():
    sizes = [20, 30, 40, 70, 10, 10]
    steps = pack_queries(sizes, range(6), max_pairs=64)
    assert steps == [[0, 1], [2], [3], [4, 5]]
    assert sorted(q for s in steps for q in s) == list(range(6))
    assert all(sum(sizes[q] for q in s) <= 64 or len(s) == 1 for s in steps)
