"""Turn hist_bench CSV output into a Markdown table."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

BASELINE = "global_atomics"


def main(path: str) -> None:
    by_dist = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_dist[r["dist"]][r["kernel"]] = r
    for dist, rows in by_dist.items():
        base = float(rows[BASELINE]["median_ms"])
        cpu = float(rows["cpu_1thread"]["median_ms"])
        print(f"\n### Input: {dist}\n")
        print("| Kernel | Median ms | Spread (p05-p95) "
              "| GB/s | vs baseline | vs CPU |")
        print("|---|---:|---:|---:|---:|---:|")
        for name, r in rows.items():
            m = float(r["median_ms"])
            spread = f"{float(r['p05_ms']):.3f}-" \
                     f"{float(r['p95_ms']):.3f}"
            print(f"| {name} | {m:.3f} | {spread} "
                  f"| {float(r['gbps']):.0f} "
                  f"| {base / m:.2f}x | {cpu / m:.1f}x |")


if __name__ == "__main__":
    main(sys.argv[1])
