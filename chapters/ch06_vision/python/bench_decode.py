import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.io import (
    ImageReadMode, decode_jpeg, read_file)

from gpubook.vision.imagefolder import (
    _resize_center, list_samples)


def timed(fn, n: int) -> float:
    fn()  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main(root: str, count: int = 256) -> None:
    _, paths, _ = list_samples(Path(root) / "train")
    paths = [p for p in paths
             if p.suffix.lower() == ".jpg"][:count]
    dev = torch.device("cuda")

    def pil_cpu():
        for p in paths:
            with Image.open(p) as im:
                im.convert("RGB").resize((256, 256))

    def tv_cpu():
        for p in paths:
            img = decode_jpeg(read_file(str(p)),
                              mode=ImageReadMode.RGB)
            _resize_center(img, 256)

    def tv_gpu():
        data = [read_file(str(p)) for p in paths]
        imgs = decode_jpeg(data, device=dev,
                           mode=ImageReadMode.RGB)
        for img in imgs:
            _resize_center(img, 256)

    n = len(paths)
    print(f"PIL, CPU          : {timed(pil_cpu, n):8.0f} img/s")
    print(f"torchvision, CPU  : {timed(tv_cpu, n):8.0f} img/s")
    print(f"torchvision, GPU  : {timed(tv_gpu, n):8.0f} img/s")


if __name__ == "__main__":
    main(sys.argv[1])
