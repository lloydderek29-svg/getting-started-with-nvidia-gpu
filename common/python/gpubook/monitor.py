"""Background GPU telemetry sampler built on NVML."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import pynvml

THROTTLE_BITS = {
    0x1: "idle",
    0x2: "app_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
    0x100: "display_clock",
}

FIELDS = [
    "t_s", "gpu_util", "mem_util", "mem_used_mib",
    "sm_mhz", "mem_mhz", "temp_c", "power_w",
    "pcie_tx_mbs", "pcie_rx_mbs", "throttle_mask",
]


def decode_throttle(mask: int) -> list[str]:
    return [n for bit, n in THROTTLE_BITS.items() if mask & bit]


def _safe(fn, default=0):
    try:
        return fn()
    except pynvml.NVMLError:
        return default


def _throttle_mask(handle) -> int:
    fn = getattr(
        pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None
    ) or getattr(
        pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons"
    )
    return int(_safe(lambda: fn(handle)))


def read_sample(handle, t0: float) -> dict:
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    tx = _safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES))
    rx = _safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES))
    return {
        "t_s": round(time.time() - t0, 3),
        "gpu_util": util.gpu,
        "mem_util": util.memory,
        "mem_used_mib": mem.used / 2**20,
        "sm_mhz": pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_SM),
        "mem_mhz": pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_MEM),
        "temp_c": pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU),
        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1e3,
        "pcie_tx_mbs": tx / 1024,  # NVML reports KB/s
        "pcie_rx_mbs": rx / 1024,
        "throttle_mask": _throttle_mask(handle),
    }


class GpuMonitor:
    """Context manager that samples one GPU in a thread."""

    def __init__(self, index=0, interval=0.5, csv_path=None):
        self.index = index
        self.interval = interval
        self.csv_path = csv_path
        self.samples: list[dict] = []
        self.marks: list[tuple[str, float]] = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._t0 = 0.0

    def __enter__(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(
            self.index)
        self._t0 = time.time()
        self._thread = threading.Thread(
            target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(
                read_sample(self._handle, self._t0))
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        pynvml.nvmlShutdown()
        if self.csv_path:
            self.write_csv(self.csv_path)

    def mark(self, label: str) -> None:
        self.marks.append((label, time.time() - self._t0))

    def write_csv(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.samples)

    def summary(self, start=None, end=None) -> dict:
        rows = [
            s for s in self.samples
            if (start is None or s["t_s"] >= start)
            and (end is None or s["t_s"] < end)
        ]
        if not rows:
            return {}

        def avg(key):
            return sum(r[key] for r in rows) / len(rows)

        mask = 0
        for r in rows:
            mask |= r["throttle_mask"]
        energy = 0.0
        for a, b in zip(rows, rows[1:]):
            dt = b["t_s"] - a["t_s"]
            energy += 0.5 * (a["power_w"] + b["power_w"]) * dt
        out = {f"{k}_avg": round(avg(k), 1) for k in (
            "gpu_util", "sm_mhz", "power_w",
            "pcie_tx_mbs", "pcie_rx_mbs")}
        out["temp_c_max"] = max(r["temp_c"] for r in rows)
        out["mem_used_mib_max"] = round(
            max(r["mem_used_mib"] for r in rows))
        out["throttle"] = decode_throttle(mask & ~0x1)
        out["energy_j"] = round(energy, 1)
        return out
