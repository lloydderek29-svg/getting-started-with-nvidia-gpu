"""Capture a snapshot of the runtime environment."""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (out.stdout + out.stderr).strip()


def installed_packages() -> dict[str, str]:
    return {
        d.metadata["Name"]: d.version
        for d in metadata.distributions()
        if d.metadata["Name"]
    }


def nvcc_version() -> str | None:
    if shutil.which("nvcc") is None:
        return None
    m = re.search(r"release (\d+\.\d+)", _run(["nvcc", "--version"]))
    return m.group(1) if m else None


def driver_version() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = _run([
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ])
    return out.splitlines()[0] if out else None


def snapshot() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "driver": driver_version(),
        "nvcc": nvcc_version(),
        "packages": dict(sorted(installed_packages().items())),
    }


def save_snapshot(path: str | Path) -> None:
    Path(path).write_text(json.dumps(snapshot(), indent=2))