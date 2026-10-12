from __future__ import annotations

import re
from collections import defaultdict

STAGES = ("queue", "compute_input", "compute_infer",
          "compute_output")

_LINE = re.compile(
    r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+(\S+)'
    r'(?:\s+\d+)?$')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

COUNTERS = {
    "ok": "nv_inference_request_success",
    "fail": "nv_inference_request_failure",
    "inferences": "nv_inference_count",
    "executions": "nv_inference_exec_count",
    "request_us": "nv_inference_request_duration_us",
    "queue_us": "nv_inference_queue_duration_us",
    "in_us": "nv_inference_compute_input_duration_us",
    "infer_us": "nv_inference_compute_infer_duration_us",
    "out_us": "nv_inference_compute_output_duration_us",
}



def _stage_ms(before, after, stage) -> float:
    b = before["inference_stats"][stage]
    a = after["inference_stats"][stage]
    count = int(a["count"]) - int(b["count"])
    ns = int(a["ns"]) - int(b["ns"])
    return ns / count / 1e6 if count > 0 else 0.0


def latency_breakdown(before: dict, after: dict) -> dict:
    """Per-request averages between two stats snapshots."""
    inferences = (int(after["inference_count"])
                  - int(before["inference_count"]))
    executions = (int(after["execution_count"])
                  - int(before["execution_count"]))
    out = {f"{s}_ms": _stage_ms(before, after, s)
           for s in STAGES}
    out["avg_batch"] = (inferences / executions
                        if executions else 0.0)
    out["requests"] = int(
        after["inference_stats"]["success"]["count"]
    ) - int(before["inference_stats"]["success"]["count"])
    return out

def parse_prometheus(text: str) -> list:
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        try:
            val = float(value)
        except ValueError:
            continue
        samples.append(
            (name, dict(_LABEL.findall(labels or "")), val))
    return samples


def total(samples, name, **match) -> float:
    return sum(v for n, lab, v in samples
               if n == name and all(
                   lab.get(k) == w for k, w in match.items()))


def per_model(samples, name) -> dict:
    out = defaultdict(float)
    for n, lab, v in samples:
        if n == name and "model" in lab:
            out[lab["model"]] += v
    return dict(out)


def _delta(before, after, name) -> dict:
    a, b = per_model(before, name), per_model(after, name)
    return {m: max(b[m] - a.get(m, 0.0), 0.0) for m in b}


def model_rates(before, after, seconds: float) -> dict:
    d = {k: _delta(before, after, n)
         for k, n in COUNTERS.items()}
    rows = {}
    for model in d["ok"].keys() | d["fail"].keys():
        v = {k: d[k].get(model, 0.0) for k in COUNTERS}
        ok = v["ok"]
        rows[model] = {
            "rps": ok / seconds,
            "fail_rps": v["fail"] / seconds,
            "avg_batch": (v["inferences"] / v["executions"]
                          if v["executions"] else 0.0),
            "latency_ms": (v["request_us"] / ok / 1e3
                           if ok else 0.0),
            "queue_ms": (v["queue_us"] / ok / 1e3
                         if ok else 0.0),
            "compute_ms": ((v["in_us"] + v["infer_us"]
                            + v["out_us"]) / ok / 1e3
                           if ok else 0.0),
        }
    return rows


def gpu_summary(samples) -> dict:
    util = [v for n, _, v in samples
            if n == "nv_gpu_utilization"]
    return {
        "util_pct": 100.0 * sum(util) / len(util)
        if util else 0.0,
        "mem_gib": total(
            samples, "nv_gpu_memory_used_bytes") / 2**30,
        "power_w": total(samples, "nv_gpu_power_usage"),
    }
