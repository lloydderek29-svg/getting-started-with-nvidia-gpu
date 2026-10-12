from __future__ import annotations

RULES = {
    "max_accuracy_gap_pts": ("accuracy_gap_pts", "max"),
    "max_p99_ms_at_target": ("p99_ms_at_target", "max"),
    "max_error_rate_at_target": ("error_rate_at_target", "max"),
    "min_capacity_rps": ("capacity_rps", "min"),
    "max_gpu_memory_gib": ("gpu_memory_gib", "max"),
}


def evaluate(metrics: dict, limits: dict,
             rules: dict = RULES) -> list[dict]:
    results = []
    for gate, limit in limits.items():
        if gate not in rules:
            raise KeyError(f"unknown objective {gate!r}")
        key, kind = rules[gate]
        if key not in metrics or metrics[key] is None:
            results.append({"gate": gate, "ok": False,
                            "value": None, "limit": limit})
            continue
        value = metrics[key]
        ok = value <= limit if kind == "max" else value >= limit
        results.append({"gate": gate, "ok": bool(ok),
                        "value": value, "limit": limit})
    return results
