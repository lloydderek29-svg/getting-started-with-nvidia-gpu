import time

import torch

from gpubook.monitor import GpuMonitor


def run_for(seconds, fn):
    end = time.time() + seconds
    while time.time() < end:
        fn()
    torch.cuda.synchronize()


def main() -> None:
    dev = torch.device("cuda")
    a = torch.randn(8192, 8192, device=dev,
                    dtype=torch.float16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    big = torch.randn(256 * 2**20, device=dev)  # 1 GiB
    host = torch.empty(512 * 2**20,
                       dtype=torch.uint8).pin_memory()
    sink = torch.empty(host.shape, dtype=torch.uint8,
                       device=dev)

    phases = [
        ("idle", 4, lambda: time.sleep(0.05)),
        ("compute", 15, lambda: torch.mm(a, b, out=c)),
        ("memory", 15, lambda: big.mul_(1.0001).add_(1e-3)),
        ("pcie", 15,
         lambda: sink.copy_(host, non_blocking=True)),
    ]

    with GpuMonitor(interval=0.5,
                    csv_path="results/ch04/monitor.csv") as mon:
        for name, seconds, fn in phases:
            mon.mark(name)
            run_for(seconds, fn)
        mon.mark("end")

    bounds = mon.marks
    for (name, t0), (_, t1) in zip(bounds, bounds[1:]):
        s = mon.summary(t0, t1)
        print(f"{name:8s} util {s['gpu_util_avg']:5.1f}%  "
              f"sm {s['sm_mhz_avg']:6.0f} MHz  "
              f"power {s['power_w_avg']:6.1f} W  "
              f"rx {s['pcie_rx_mbs_avg']:8.0f} MB/s  "
              f"max {s['temp_c_max']} C  "
              f"{s['throttle']}")


if __name__ == "__main__":
    main()
