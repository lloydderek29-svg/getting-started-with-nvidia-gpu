from __future__ import annotations

from PIL import Image, ImageDraw


def make_grid(images, labels, cols: int = 3):
    w, h = images[0].size
    rows = -(-len(images) // cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(zip(images, labels)):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(img, (x, y))
        draw.text((x + 8, y + 8), label, fill="white")
    return sheet
