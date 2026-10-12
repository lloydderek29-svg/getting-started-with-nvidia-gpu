import torch

from finetune import top_confusions

CLASSES = ["cat", "dog", "fox"]


def test_top_confusions_counts_errors_only():
    labels = torch.tensor([0, 0, 0, 1, 2])
    preds = torch.tensor([1, 1, 0, 1, 1])
    result = top_confusions(preds, labels, CLASSES, k=2)
    assert result[0] == ("cat", "dog", 2)
    assert result[1] == ("fox", "dog", 1)
