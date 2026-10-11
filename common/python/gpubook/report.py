"""Build a GPU capability report."""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
from pathlib import Path

SMI_FIELDS = [
    "index",
    "name",
    "compute_cap",
    "driver_version",
    "memory.total",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "power.limit",
    "clocks.max.sm",
    "clocks.max.mem",
]


def parse_smi_csv(text: str) -> list[dict]:
    """Convert nvidia-smi CSV output to a list of dicts."""
    rows = []
    reader = csv.reader(io.StringIO(text),
                        skipinitialspace=True)
    for values in reader:
        if values:
            rows.append(dict(zip(SMI_FIELDS, values)))
    return rows


def query_smi() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi not found on PATH")
    cmd = [
        exe,
        "--query-gpu=" + ",".join(SMI_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True
    )
    return parse_smi_csv(result.stdout)


def query_runtime(binary: Path) -> list[dict]:
    if not binary.exists():
        return []
    result = subprocess.run(
        [str(binary)], capture_output=True,
        text=True, check=True,
    )
    return [json.loads(line)
            for line in result.stdout.splitlines()
            if line.strip()]


def decode_version(v: int) -> str:
    return f"{v // 1000}.{(v % 1000) // 10}"


def derive_notes(gpu: dict) -> list[str]:
    """Plain-language warnings based on collected data."""
    notes = []
    width = gpu.get("pcie.link.width.current", "")
    if width.isdigit() and int(width) < 16:
        notes.append(
            f"PCIe link is x{width} (x16 expected; "
            "the link may drop its width when idle)"
        )
    rt = gpu.get("runtime") or {}
    if rt and rt.get("runtime_version", 0) > rt.get(
        "driver_version", 0
    ):
        notes.append(
            "CUDA runtime is newer than the driver supports"
        )
    return notes


def build_report(binary: Path) -> dict:
    smi = query_smi()
    runtime = {r["index"]: r for r in query_runtime(binary)}
    gpus = []
    for row in smi:
        row = dict(row)
        row["runtime"] = runtime.get(int(row["index"]))
        row["notes"] = derive_notes(row)
        gpus.append(row)
    return {"gpus": gpus}


def format_report(report: dict) -> str:
    lines = []
    for g in report["gpus"]:
        lines.append(f"GPU {g['index']}: {g['name']}")
        lines.append(f"  Compute capability : {g['compute_cap']}")
        lines.append(f"  Driver             : "
                     f"{g['driver_version']}")
        lines.append(f"  Memory (MiB)       : "
                     f"{g['memory.total']}")
        lines.append(
            f"  PCIe               : Gen "
            f"{g['pcie.link.gen.current']} "
            f"x{g['pcie.link.width.current']}"
        )
        lines.append(f"  Power limit (W)    : "
                     f"{g['power.limit']}")
        rt = g.get("runtime")
        if rt:
            lines.append(f"  SM count           : "
                         f"{rt['sm_count']}")
            lines.append(f"  Peak bandwidth     : "
                         f"{rt['peak_bw_gbs']} GB/s")
            lines.append(
                "  CUDA runtime       : "
                f"{decode_version(rt['runtime_version'])}"
            )
        for note in g["notes"]:
            lines.append(f"  NOTE: {note}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runtime",
        default="build/release/bin/runtime_report",
        help="path to the compiled runtime_report tool",
    )
    ap.add_argument("--json", help="write report to file")
    args = ap.parse_args()

    report = build_report(Path(args.runtime))
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2)
        )
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
