"""Run ncu on a program and print a compact kernel table."""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import sys

METRICS = {
    "gpu__time_duration.sum": "time_us",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":
        "sm_pct",
    "gpu__compute_memory_throughput.avg."
    "pct_of_peak_sustained_elapsed": "mem_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":
        "dram_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active":
        "warps_pct",
    "launch__registers_per_thread": "regs",
}


def find_ncu() -> str:
    exe = shutil.which("ncu")
    if exe:
        return exe
    return "/usr/local/cuda/bin/ncu"


def run_ncu(kernel_regex, count, cmd) -> list[dict]:
    args = [
        find_ncu(), "--csv", "--page", "raw",
        "--print-units", "base",
        "--metrics", ",".join(METRICS),
        "--kernel-name", f"regex:{kernel_regex}",
        "--launch-count", str(count),
    ] + cmd
    out = subprocess.run(
        args, capture_output=True, text=True, check=True
    ).stdout
    # ncu mixes status lines with CSV; keep quoted rows only.
    lines = [ln for ln in out.splitlines()
             if ln.startswith('"')]
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    return [r for r in reader
            if r.get("ID", "").strip().isdigit()]


def num(text: str) -> float:
    return float(text.replace(",", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=".*")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("cmd", nargs="+")
    a = ap.parse_args()

    rows = run_ncu(a.kernel, a.count, a.cmd)
    head = (f"{'kernel':26s} {'us':>9s} {'SM%':>6s} "
            f"{'Mem%':>6s} {'DRAM%':>6s} {'Warps%':>7s} "
            f"{'Regs':>5s}")
    print(head)
    for r in rows:
        v = {short: num(r[long])
             for long, short in METRICS.items()}
        name = r["Kernel Name"].split("(")[0][-26:]
        print(f"{name:26s} {v['time_us'] / 1e3:9.1f} "
              f"{v['sm_pct']:6.1f} {v['mem_pct']:6.1f} "
              f"{v['dram_pct']:6.1f} {v['warps_pct']:7.1f} "
              f"{v['regs']:5.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
