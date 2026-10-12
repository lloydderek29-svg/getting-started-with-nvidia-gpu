import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root: Path) -> list[dict]:
    rows = []
    for split in sorted(p for p in root.iterdir() if p.is_dir()):
        for cls in sorted(p for p in split.iterdir()
                          if p.is_dir()):
            for f in sorted(cls.rglob("*")):
                if f.suffix.lower() not in EXTS:
                    continue
                row = {
                    "split": split.name, "cls": cls.name,
                    "path": str(f), "sha1": file_hash(f),
                    "w": 0, "h": 0, "ok": True,
                }
                try:
                    with Image.open(f) as im:
                        im.verify()
                    with Image.open(f) as im:
                        row["w"], row["h"] = im.size
                except Exception:
                    row["ok"] = False
                rows.append(row)
    return rows


def find_duplicates(rows: list[dict]) -> list[list[str]]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["sha1"]].append(r["path"])
    return [p for p in groups.values() if len(p) > 1]


def find_leakage(rows: list[dict]) -> list[str]:
    splits = defaultdict(set)
    for r in rows:
        splits[r["sha1"]].add(r["split"])
    return [h for h, s in splits.items() if len(s) > 1]


def class_counts(rows: list[dict], split: str) -> Counter:
    return Counter(r["cls"] for r in rows
                   if r["split"] == split and r["ok"])


def imbalance_ratio(counts: Counter) -> float:
    if not counts:
        return 1.0
    return max(counts.values()) / max(min(counts.values()), 1)


def main(root: str) -> int:
    rows = scan(Path(root))
    bad = [r["path"] for r in rows if not r["ok"]]
    dups = find_duplicates(rows)
    leaks = find_leakage(rows)
    print(f"images scanned : {len(rows)}")
    print(f"corrupt files  : {len(bad)}")
    print(f"duplicate sets : {len(dups)}")
    print(f"cross-split    : {len(leaks)}")
    for split in sorted({r["split"] for r in rows}):
        c = class_counts(rows, split)
        print(f"{split:6s} classes={len(c)} "
              f"imbalance={imbalance_ratio(c):.1f}x")
    out = Path(root) / "manifest.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"manifest written to {out}")
    return 1 if (bad or leaks) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
