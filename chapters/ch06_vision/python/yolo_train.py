from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def metrics_dict(m) -> dict:
    return {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "speed_ms": {k: float(v) for k, v in m.speed.items()},
    }


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        seed=cfg["seed"],
        workers=cfg["workers"],
        cache=cfg["cache"],
        device=0,
        project=cfg["project"],
        name=cfg["name"],
        exist_ok=True,
    )
    m = model.val(data=cfg["data"], imgsz=cfg["imgsz"])
    out = Path(cfg["project"]) / cfg["name"] / "summary.json"
    out.write_text(json.dumps(metrics_dict(m), indent=2))
    print(json.dumps(metrics_dict(m), indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
