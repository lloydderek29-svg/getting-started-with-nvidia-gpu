from collections import Counter

from audit_dataset import (
    find_duplicates, find_leakage, imbalance_ratio)

ROWS = [
    {"split": "train", "cls": "a", "path": "t/a/1", "sha1": "x"},
    {"split": "train", "cls": "a", "path": "t/a/2", "sha1": "x"},
    {"split": "val", "cls": "a", "path": "v/a/1", "sha1": "x"},
    {"split": "val", "cls": "b", "path": "v/b/1", "sha1": "y"},
]


def test_duplicates_are_grouped():
    assert find_duplicates(ROWS) == [["t/a/1", "t/a/2", "v/a/1"]]


def test_leakage_detects_shared_hash():
    assert find_leakage(ROWS) == ["x"]


def test_imbalance_ratio():
    assert imbalance_ratio(Counter({"a": 90, "b": 30})) == 3.0
