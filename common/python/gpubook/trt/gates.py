from __future__ import annotations

# gate key -> (metric key, kind) where kind is "max" or "min"
RULES = {
    "max_accuracy_drop_pts": ("accuracy_drop_pts", "max"),
    "min_top1_agreement": ("top1_agreement", "min"),
    "max_p99_ms_b1": ("p99_ms_b1", "max"),
    "min_throughput_b32": ("throughput_b32", "min"),
    "max_memory_gib": ("memory_gib", "max"),
    "max_latency_drift": ("latency_drift", "max"),
    "max_memory_growth_mib": ("memory_growth_mib", "max"),
    "max_batch_invariance_diff": ("batch_invariance_diff", "max"),
}


def evaluate_gates(metrics: dict, gates: dict) -> list[dict]:
    results = []
    for gate, limit in gates.items():
        if gate not in RULES:
            raise KeyError(f"unknown gate {gate!r}")
        key, kind = RULES[gate]
        if key not in metrics:
            results.append({"gate": gate, "ok": False,
                            "value": None, "limit": limit,
                            "note": "metric missing"})
            continue
        value = metrics[key]
        ok = value <= limit if kind == "max" else value >= limit
        results.append({"gate": gate, "ok": bool(ok),
                        "value": value, "limit": limit,
                        "note": ""})
    return results
