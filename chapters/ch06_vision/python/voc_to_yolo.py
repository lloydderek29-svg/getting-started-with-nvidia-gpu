from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def voc_box_to_yolo(xmin, ymin, xmax, ymax, w, h):
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return cx, cy, bw, bh


def convert_file(xml_path: Path, classes: list[str]) -> str:
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in classes:
            continue
        box = obj.find("bndbox")
        vals = [float(box.findtext(k)) for k in
                ("xmin", "ymin", "xmax", "ymax")]
        cx, cy, bw, bh = voc_box_to_yolo(*vals, w, h)
        lines.append(
            f"{classes.index(name)} "
            f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def main(xml_dir: str, out_dir: str, class_file: str) -> None:
    classes = Path(class_file).read_text().split()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for xml in sorted(Path(xml_dir).glob("*.xml")):
        text = convert_file(xml, classes)
        (out / f"{xml.stem}.txt").write_text(text)


if __name__ == "__main__":
    main(*sys.argv[1:4])
# chapters/ch06_vision/tests/test_voc_to_yolo.py
from voc_to_yolo import voc_box_to_yolo


def test_box_conversion_is_normalized_center_format():
    box = voc_box_to_yolo(0, 0, 100, 50, 200, 100)
    assert box == (0.25, 0.25, 0.5, 0.5)
