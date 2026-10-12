from __future__ import annotations

import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_label_text(text: str, num_classes: int):
    problems = []
    for n, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"line {n}: expected 5 fields")
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            problems.append(f"line {n}: non-numeric value")
            continue
        if not 0 <= cls < num_classes:
            problems.append(f"line {n}: class {cls} invalid")
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            problems.append(f"line {n}: value outside 0..1")
        if w <= 0 or h <= 0:
            problems.append(f"line {n}: empty box")
    return problems


def check_split(root: Path, split: str, num_classes: int):
    bad = 0
    images = [p for p in sorted(
        (root / "images" / split).rglob("*"))
        if p.suffix.lower() in IMG_EXTS]
    for img in images:
        label = root / "labels" / split / f"{img.stem}.txt"
        if not label.exists():
            print(f"missing label: {img}")
            bad += 1
            continue
        for msg in validate_label_text(
                label.read_text(), num_classes):
            print(f"{label}: {msg}")
            bad += 1
    print(f"{split}: {len(images)} images, {bad} problems")
    return bad


if __name__ == "__main__":
    root, nc = Path(sys.argv[1]), int(sys.argv[2])
    total = sum(check_split(root, s, nc)
                for s in ("train", "val"))
    sys.exit(1 if total else 0)
# chapters/ch06_vision/tests/test_check_yolo.py
from check_yolo_dataset import validate_label_text


def test_valid_label_has_no_problems():
    assert validate_label_text("2 0.5 0.5 0.2 0.3", 4) == []


def test_bad_class_and_range_are_reported():
    text = "9 0.5 0.5 0.2 0.3\n1 1.5 0.5 0.2 0.3"
    problems = validate_label_text(text, 4)
    assert any("class 9" in p for p in problems)
    assert any("outside" in p for p in problems)


def test_wrong_field_count_is_reported():
    assert validate_label_text("1 0.5 0.5", 4)
