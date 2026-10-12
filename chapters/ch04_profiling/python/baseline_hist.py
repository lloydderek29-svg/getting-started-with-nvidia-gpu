import numpy as np
import torch

from gpubook.bench import bench_cpu, bench_cuda


def main() -> None:
    n = 64 * 2**20
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, n, dtype=np.uint8)

    cpu = bench_cpu(
        lambda: np.bincount(data, minlength=256),
        warmup=2,
        repeats=10,
    )
    print(f"NumPy bincount      : {cpu.median_ms:8.3f} ms")

    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    gpu_data = torch.from_numpy(data).to(dev).to(torch.int32)
    kernel = bench_cuda(
        lambda: torch.bincount(gpu_data, minlength=256),
        flush_l2_mb=256,
    )
    print(f"torch.bincount (GPU): {kernel.median_ms:8.3f} ms")

    host = torch.from_numpy(data)
    pinned = host.pin_memory()
    pageable = bench_cuda(lambda: host.to(dev))
    pin = bench_cuda(
        lambda: pinned.to(dev, non_blocking=True)
    )
    print(f"Copy in, pageable   : {pageable.median_ms:8.3f} ms")
    print(f"Copy in, pinned     : {pin.median_ms:8.3f} ms")


if __name__ == "__main__":
    main()
