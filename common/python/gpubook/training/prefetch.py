from __future__ import annotations

import torch


class CudaPrefetcher:
    """Copy the next batch to the GPU on a side stream."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __len__(self):
        return len(self.loader)

    def _preload(self, it):
        try:
            x, y = next(it)
        except StopIteration:
            return None
        with torch.cuda.stream(self.stream):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y

    def __iter__(self):
        it = iter(self.loader)
        batch = self._preload(it)
        current = torch.cuda.current_stream()
        while batch is not None:
            current.wait_stream(self.stream)
            x, y = batch
            # Tell the allocator these tensors are used on the
            # compute stream so their memory is not reused early.
            x.record_stream(current)
            y.record_stream(current)
            batch = self._preload(it)
            yield x, y
