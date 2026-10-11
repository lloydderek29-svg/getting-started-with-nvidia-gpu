import torch


def measure_peak(model, batch, steps: int = 3) -> float:
    """Return peak allocated GiB for a training step."""
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model(batch).float().mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30
