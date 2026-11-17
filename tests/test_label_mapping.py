import numpy as np
import pytest

from src.judge import SYSTEM, expected_gain, label_token_ids
from src.metrics import GAINS, LABELS


class FakeTokenizer:
    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=False):
        return self.table[text]


def test_label_order_and_gains():
    assert LABELS == ("E", "S", "C", "I")
    one_hot = np.eye(4)
    assert expected_gain(one_hot).tolist() == [GAINS[label] for label in LABELS] == [1.0, 0.1, 0.01, 0.0]


def test_expected_gain_is_probability_weighted():
    probs = np.array([[0.5, 0.3, 0.1, 0.1], [0.0, 0.0, 0.0, 1.0]])
    assert expected_gain(probs) == pytest.approx([0.5 + 0.03 + 0.001, 0.0])


def test_label_token_ids_follow_label_order():
    tokenizer = FakeTokenizer({"E": [36], "S": [50], "C": [34], "I": [40]})
    assert label_token_ids(tokenizer) == [36, 50, 34, 40]


def test_multi_token_or_duplicate_labels_are_rejected():
    with pytest.raises(RuntimeError):
        label_token_ids(FakeTokenizer({"E": [1, 2], "S": [3], "C": [4], "I": [5]}))
    with pytest.raises(RuntimeError):
        label_token_ids(FakeTokenizer({"E": [1], "S": [1], "C": [4], "I": [5]}))


def test_prompt_defines_every_label():
    for label, word in zip(LABELS, ("Exact", "Substitute", "Complement", "Irrelevant")):
        assert f"{label} ({word})" in SYSTEM
