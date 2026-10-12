import argparse
import json
from pathlib import Path

from gpubook.llm.loading import (
    GIB, compat_report, fetch_model, fits, inventory)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    a = ap.parse_args()

    path = fetch_model(a.model)
    inv = inventory(path)
    cfg = json.loads((Path(path) / "config.json").read_text())
    rep = compat_report()

    print(f"model        : {a.model}")
    print(f"parameters   : {inv['params'] / 1e9:.2f} B")
    print(f"stored as    : {inv['bytes'] / GIB:.2f} GiB "
          f"{inv['dtypes']}")
    print(f"max context  : "
          f"{cfg.get('max_position_embeddings', 'unknown')}")
    for key, value in rep.items():
        print(f"{key:20s}: {value}")

    print("\nverdict for weights alone plus a 1.5 GiB reserve")
    for bits in (16, 8, 4):
        need = inv["params"] * bits / 8 / GIB
        ok = fits(need, rep["free_gib"])
        print(f"  {bits:2d}-bit: {need:6.2f} GiB  "
              f"{'fits' if ok else 'does not fit'}")


if __name__ == "__main__":
    main()
python chapters/ch07_llm/python/preflight.py \
    Qwen/Qwen2.5-1.5B-Instruct
