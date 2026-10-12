import copy

import torch


def test_accumulation_matches_large_batch():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    ref = copy.deepcopy(model)
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    mse = torch.nn.MSELoss()

    mse(ref(x), y).backward()

    for k in range(4):
        xs = x[2 * k:2 * k + 2]
        ys = y[2 * k:2 * k + 2]
        (mse(model(xs), ys) / 4).backward()

    assert torch.allclose(
        model.weight.grad, ref.weight.grad, atol=1e-6)
