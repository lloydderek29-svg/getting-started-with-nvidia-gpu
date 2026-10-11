"""Validate the GPU software stack."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return p.returncode, (p.stdout + p.stderr).strip()


def parse_cuda_version(text: str) -> tuple[int, int] | None:
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def check_driver() -> tuple[Check, tuple[int, int] | None]:
    if shutil.which("nvidia-smi") is None:
        return Check("driver", False, "nvidia-smi missing"), None
    code, out = run(["nvidia-smi"])
    if code != 0:
        first = out.splitlines()[0] if out else "no output"
        return Check("driver", False, first), None
    ver = parse_cuda_version(out)
    detail = f"supports CUDA {ver[0]}.{ver[1]}" if ver else ""
    return Check("driver", True, detail), ver


def check_nvcc() -> Check:
    if shutil.which("nvcc") is None:
        return Check("nvcc", False, "not on PATH")
    _, out = run(["nvcc", "--version"])
    m = re.search(r"release (\d+\.\d+)", out)
    return Check("nvcc", True, f"release {m.group(1)}" if m else "")


def check_torch(driver_cuda) -> Check:
    try:
        import torch
    except ImportError:
        return Check("torch", False, "not installed")
    if not torch.cuda.is_available():
        return Check("torch", False, "CUDA not available")
    built = torch.version.cuda or ""
    detail = f"built for CUDA {built}"
    if driver_cuda and built:
        major = int(built.split(".")[0])
        if major > driver_cuda[0]:
            return Check("torch", False,
                         detail + "; driver is too old")
    return Check("torch", True, detail)


def run_all() -> list[Check]:
    driver, ver = check_driver()
    return [driver, check_nvcc(), check_torch(ver)]


def main() -> int:
    checks = run_all()
    for c in checks:
        tag = "OK  " if c.ok else "FAIL"
        print(f"[{tag}] {c.name:8s} {c.detail}")
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
