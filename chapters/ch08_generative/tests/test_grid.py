from PIL import Image

from sched_compare import make_grid


def test_grid_dimensions():
    imgs = [Image.new("RGB", (32, 32)) for _ in range(5)]
    sheet = make_grid(imgs, list("abcde"), cols=3)
    assert sheet.size == (96, 64)
